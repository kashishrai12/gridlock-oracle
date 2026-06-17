"""
find_high_risk_example.py — find REAL closure events the model scores highest, using the
SAME fields the (restructured) Predict Event page collects, so the printed probability
reproduces EXACTLY when you type the values into the dashboard.

Fields the UI uses: event_cause, veh_type, priority, event_type, time-of-day,
police_station, and (optional) junction. Zone/corridor are NOT used by the closure model
and are omitted. Blank vehicle/junction are sent as "none" (the "— not specified —"
dropdown option).

Run: python find_high_risk_example.py --data data/flipkart_gridlock.csv
"""

import argparse
import pandas as pd
from predictor import GridlockPredictor

NOT_SET = "— not specified —"
# fields the model/UI actually use
OPT_BLANK = {"veh_type", "junction"}            # have a "not specified" option in the UI


def _clean(v):
    s = str(v).strip()
    return "none" if s.lower() in ("", "nan", "none") else s


def build_event(row):
    """Mirror the dashboard's event dict exactly (fixed date + row's hour)."""
    ts = pd.to_datetime(row.get("start_datetime"), errors="coerce")
    hour = int(ts.hour) if pd.notna(ts) else 18
    return {
        "start_datetime": f"2024-06-01 {hour:02d}:00",
        "event_cause": _clean(row.get("event_cause")),
        "veh_type": _clean(row.get("veh_type")),
        "police_station": _clean(row.get("police_station")),
        "junction": _clean(row.get("junction")),
        "priority": _clean(row.get("priority")) if _clean(row.get("priority")) != "none" else "Low",
        "event_type": _clean(row.get("event_type")) if _clean(row.get("event_type")) != "none" else "unplanned",
    }, hour


def _disp(v):
    return NOT_SET if str(v).lower() == "none" else v


def main(path, n=5):
    df = pd.read_csv(path)
    predictor = GridlockPredictor()
    mask = df["requires_road_closure"].astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])
    pos = df[mask].copy()

    sample = pos.sample(min(400, len(pos)), random_state=0)
    scored = []
    for _, row in sample.iterrows():
        e, hour = build_event(row.to_dict())
        scored.append((predictor.predict(e)["closure_prob"], e, hour))

    seen, uniq = set(), []
    for prob, e, hour in sorted(scored, key=lambda x: x[0], reverse=True):
        key = (e["event_cause"], e["veh_type"], e["police_station"], e["junction"],
               e["priority"], e["event_type"], hour)
        if key in seen:
            continue
        seen.add(key); uniq.append((prob, e, hour))

    print(f"\nTop {n} real CLOSURE events the model scores highest.")
    print("Enter these EXACT values in the Predict Event page to reproduce the probability:\n")
    for i, (prob, e, hour) in enumerate(uniq[:n], 1):
        print(f"════════ Example {i} — predicted closure prob: {prob:.1%} ════════")
        print(f"   Event cause     : {e['event_cause']}")
        print(f"   Vehicle type    : {_disp(e['veh_type'])}")
        print(f"   Priority        : {e['priority']}")
        print(f"   Type            : {e['event_type']}")
        print(f"   Time of day     : {hour:02d}:00")
        print(f"   Police station  : {e['police_station']}")
        print(f"   Junction (opt.) : {_disp(e['junction'])}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)