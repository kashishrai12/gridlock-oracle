"""
hotspots.py  —  spatiotemporal hotspot analytics (the "manpower" pillar)

Descriptive, not predictive: aggregates the full event log to show WHERE and WHEN
incidents cluster, and converts expected load into a deployment recommendation.
Uses ALL events with a valid start time (not just closure-labelled ones).

Run: python hotspots.py --data data/flipkart_gridlock.csv
Outputs (models/):
    hotspot_junctions.csv   top junctions by event load (+ closure rate)
    hotspot_zone_hour.csv   zone x hour-of-day matrix (heatmap source)
    hotspot_hourly.csv      city-wide hourly profile
    hotspot_dow.csv         day-of-week profile
API: HotspotAnalyzer(models_dir).recommend(zone, hour, day_of_week)
"""

import argparse, os
import numpy as np
import pandas as pd

MODELS_DIR = "models"


def _load_all_events(path):
    df = pd.read_csv(path)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df = df[df["start_datetime"].notna()].copy()
    df["hour"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek          # 0=Mon
    df["date"] = df["start_datetime"].dt.date
    if "requires_road_closure" in df.columns:
        df["closure_int"] = (df["requires_road_closure"].astype(str).str.lower()
                             .isin(["1", "true", "yes", "y", "t"]).astype(int))
    else:
        df["closure_int"] = 0
    return df


def build(path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = _load_all_events(path)
    n_days = max(df["date"].nunique(), 1)
    print(f"[hotspots] {len(df)} events over {n_days} days")

    # --- top junctions by load ---
    if "junction" in df.columns:
        j = (df.groupby("junction")
               .agg(events=("id", "size"), closure_rate=("closure_int", "mean"))
               .sort_values("events", ascending=False))
        j["events_per_week"] = (j["events"] / n_days * 7).round(1)
        j.reset_index().to_csv(f"{MODELS_DIR}/hotspot_junctions.csv", index=False)
        print(f"[hotspots] top junction: {j.index[0]} ({int(j['events'].iloc[0])} events)")

    # --- zone x hour matrix (heatmap) ---
    zkey = "zone" if "zone" in df.columns else None
    if zkey:
        zh = df.pivot_table(index=zkey, columns="hour", values="id",
                            aggfunc="size", fill_value=0)
        zh = zh.reindex(columns=range(24), fill_value=0)
        zh.reset_index().to_csv(f"{MODELS_DIR}/hotspot_zone_hour.csv", index=False)

    # --- temporal profiles ---
    (df.groupby("hour").size().reindex(range(24), fill_value=0)
       .rename("events").reset_index().to_csv(f"{MODELS_DIR}/hotspot_hourly.csv", index=False))
    (df.groupby("day_of_week").size().reindex(range(7), fill_value=0)
       .rename("events").reset_index().to_csv(f"{MODELS_DIR}/hotspot_dow.csv", index=False))

    peak_hour = int(df.groupby("hour").size().idxmax())
    print(f"[hotspots] city-wide peak hour: {peak_hour:02d}:00")
    print(f"[hotspots] artifacts written to {MODELS_DIR}/")
    return df


class HotspotAnalyzer:
    """Loads the zone x hour matrix and turns expected load into a deployment hint."""
    def __init__(self, models_dir=MODELS_DIR):
        self.zh = pd.read_csv(f"{models_dir}/hotspot_zone_hour.csv")
        self.zcol = self.zh.columns[0]
        self.hour_cols = [c for c in self.zh.columns if str(c).isdigit()]
        # baseline: average events per zone-hour cell, for normalising load
        self.cell_mean = self.zh[self.hour_cols].values.mean() or 1.0

    def expected_load(self, zone, hour):
        r = self.zh[self.zh[self.zcol] == zone]
        if r.empty or str(hour) not in self.zh.columns:
            return 0.0
        return float(r[str(hour)].iloc[0])

    def recommend(self, zone, hour, day_of_week=None):
        """Map expected event load at a zone-hour to a suggested officer count.
        Transparent rule: more historical load -> more pre-positioned manpower."""
        load = self.expected_load(zone, hour)
        ratio = load / self.cell_mean
        officers = int(np.clip(round(2 + 2 * ratio), 2, 20))
        level = "HIGH" if ratio >= 2 else "MODERATE" if ratio >= 1 else "LOW"
        return {"zone": zone, "hour": hour, "expected_events": round(load, 1),
                "load_level": level, "suggested_officers": officers}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    df = build(ap.parse_args().data)
    # quick demo of the deployment recommender
    try:
        a = HotspotAnalyzer()
        top_zone = df["zone"].value_counts().idxmax() if "zone" in df else None
        peak = int(df.groupby("hour").size().idxmax())
        if top_zone is not None:
            import json
            print("\n[demo] deployment recommendation:")
            print(json.dumps(a.recommend(top_zone, peak), indent=2))
    except Exception as e:
        print(f"[demo] skipped ({e})")