"""
live_feed.py — LIVE incident feed.

"Live" = incidents arrive from a stream (not a form) and are scored automatically by the
existing models. Two sources:

  1. REAL API (TomTom Traffic Incidents) — if TOMTOM_API_KEY is set, pulls REAL incidents
     happening in Bengaluru right now (accidents, jams, closures, flooding, breakdowns).
     Mappls does not offer a city-wide incident feed, so we use TomTom for this.
  2. REPLAY (always works) — replays a real high-incident day from the dataset as a
     fast-forward live stream. Honest and always dense, so the demo never looks empty.

Each incident is scored by the closure predictor and folded into the Hawkes cascade model so
the live citywide cascade risk updates as incidents arrive.

Get a free key (2,500 req/day) at https://developer.tomtom.com -> then:
    setx TOMTOM_API_KEY "your_key"
"""

import os
import pandas as pd
import numpy as np

# Bengaluru bounding box (minLon, minLat, maxLon, maxLat)
BENGALURU_BBOX = (77.46, 12.83, 77.78, 13.14)
BLR_CENTER = (12.9716, 77.5946)

# TomTom iconCategory -> your model's event_cause vocabulary (best-effort alignment)
ICON_TO_CAUSE = {
    0: "others", 1: "accident", 2: "others", 3: "others", 4: "water_logging",
    5: "others", 6: "others", 7: "others", 8: "others", 9: "construction",
    10: "others", 11: "water_logging", 14: "vehicle_breakdown",
}
ICON_LABEL = {
    1: "Accident", 6: "Traffic jam", 7: "Lane closed", 8: "Road closed",
    9: "Road works", 11: "Flooding", 14: "Broken-down vehicle",
}


# --------------------------------------------------------------------------- #
# REAL API source — TomTom Traffic Incidents v5
# --------------------------------------------------------------------------- #
def source_status():
    return "live-api" if os.environ.get("TOMTOM_API_KEY", "").strip() else "replay"


def real_feed_incidents(bbox=BENGALURU_BBOX):
    """Return list of live incidents [{cause,label,lat,lon,location,severity,minutes_ago}] or None."""
    key = os.environ.get("TOMTOM_API_KEY", "").strip()
    if not key:
        return None
    fields = ("{incidents{type,geometry{type,coordinates},"
              "properties{iconCategory,magnitudeOfDelay,startTime,"
              "events{description,code,iconCategory},from,to,delay,length}}}")
    try:
        import requests
        r = requests.get(
            "https://api.tomtom.com/traffic/services/5/incidentDetails",
            params={"key": key,
                    "bbox": ",".join(str(b) for b in bbox),
                    "fields": fields,
                    "language": "en-GB",
                    "timeValidityFilter": "present"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        now = pd.Timestamp.utcnow()
        out = []
        for f in data.get("incidents", []):
            props = f.get("properties", {}) or {}
            geom = f.get("geometry", {}) or {}
            coords = geom.get("coordinates") or []
            if coords and isinstance(coords[0], (list, tuple)):
                lon, lat = coords[0][0], coords[0][1]
            elif len(coords) >= 2:
                lon, lat = coords[0], coords[1]
            else:
                continue
            icon = props.get("iconCategory", 0)
            events = props.get("events") or []
            desc = events[0].get("description") if events else ICON_LABEL.get(icon, "Incident")
            start = props.get("startTime")
            mins_ago = 0.0
            if start:
                try:
                    mins_ago = max(0.0, (now - pd.to_datetime(start, utc=True)).total_seconds() / 60.0)
                except Exception:
                    mins_ago = 0.0
            out.append({
                "cause": ICON_TO_CAUSE.get(icon, "others"),
                "label": ICON_LABEL.get(icon, desc or "Incident"),
                "lat": float(lat), "lon": float(lon),
                "location": (props.get("from") or desc or "live"),
                "severity": props.get("magnitudeOfDelay", 0),
                "minutes_ago": mins_ago,
            })
        return out
    except Exception:
        return None


def live_scored(predictor, bbox=BENGALURU_BBOX):
    """Triage live incidents using what TomTom actually provides (category + severity).
    The closure model needs features TomTom doesn't supply, so we triage on real signal."""
    inc = real_feed_incidents(bbox)
    if not inc:
        return None

    # readiness from TomTom magnitudeOfDelay (severity) + category
    HIGH_CATS = {"Road closed", "Accident", "Flooding"}       # closure-likely
    MED_CATS = {"Lane closed", "Broken-down vehicle", "Road works"}

    rows = []
    for a in inc:
        sev = int(a.get("severity") or 0)        # 0..4 (TomTom magnitudeOfDelay)
        label = a["label"]
        # tier logic: severe delay OR closure-type -> pre-position; moderate -> standby; else monitor
        if label in {"Road closed", "Accident", "Flooding"} or sev >= 4:
            tier, score = "PRE-POSITION", 0.65
        elif label in {"Lane closed", "Broken-down vehicle", "Road works"} or sev == 3:
            tier, score = "STANDBY", 0.3
        else:
            tier, score = "MONITOR", 0.12
        rows.append({"t_min": -a["minutes_ago"], "time": "now",
                     "cause": label, "vehicle": "—", "location": a["location"],
                     "lat": a["lat"], "lon": a["lon"],
                     "closure_prob": score, "impact": round(2 + sev * 1.8, 1),
                     "readiness": tier})
    return pd.DataFrame(rows).sort_values("closure_prob", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------------------- #
# REPLAY source (always works)
# --------------------------------------------------------------------------- #
def prepare_replay(path, predictor, max_events=80):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df = df[ts.notna()].copy(); ts = ts[ts.notna()]
    df["ts"] = ts.values
    df["date"] = ts.dt.date.values
    busiest = df["date"].value_counts().idxmax()
    day = df[df["date"] == busiest].sort_values("ts").head(max_events).reset_index(drop=True)
    t0 = day["ts"].min()
    day["t_min"] = (day["ts"] - t0).dt.total_seconds() / 60.0
    has_geo = {"latitude", "longitude"}.issubset(day.columns)

    rows = []
    for _, r in day.iterrows():
        ev = {"start_datetime": str(r["ts"]),
              "event_cause": str(r.get("event_cause", "none")),
              "veh_type": str(r.get("veh_type", "none")),
              "priority": str(r.get("priority", "Medium")),
              "event_type": str(r.get("event_type", "unplanned")),
              "police_station": str(r.get("police_station", "none")),
              "junction": str(r.get("junction", "none"))}
        try:
            pr = predictor.predict(ev)
            cp, imp, tier = pr["closure_prob"], pr["impact_score"], pr.get("readiness_tier", "")
        except Exception:
            cp, imp, tier = 0.1, 3.0, "MONITOR"
        rows.append({
            "t_min": float(r["t_min"]),
            "time": pd.to_datetime(r["ts"]).strftime("%H:%M"),
            "cause": ev["event_cause"], "vehicle": ev["veh_type"],
            "location": str(r.get("corridor") if str(r.get("corridor", "nan")) not in ("nan", "Non-corridor")
                            else r.get("junction", "unknown")),
            "lat": float(r["latitude"]) if has_geo and pd.notna(r.get("latitude")) else None,
            "lon": float(r["longitude"]) if has_geo and pd.notna(r.get("longitude")) else None,
            "closure_prob": cp, "impact": imp, "readiness": tier,
        })
    return pd.DataFrame(rows), str(busiest)


def active_incidents(replay_df, sim_now_min, lookback_min=45):
    m = (replay_df["t_min"] <= sim_now_min) & (replay_df["t_min"] > sim_now_min - lookback_min)
    return replay_df[m].sort_values("t_min", ascending=False)


def live_cascade_risk(active_df, sim_now_min, hawkes_model):
    """Expected follow-on incidents in the next 60 min from currently-active incidents."""
    if hawkes_model is None or active_df.empty:
        return 0.0
    # each active incident contributes branching * decay toward future follow-ons
    import math
    alpha = hawkes_model.p["alpha"]; beta = hawkes_model.p["beta"]
    total = 0.0
    for _, t in (sim_now_min - active_df["t_min"]).clip(lower=0).items():
        # remaining excitation it will still produce over next 60 min
        total += (alpha / beta) * math.exp(-beta * t) * (1 - math.exp(-beta * 60))
    return float(total)