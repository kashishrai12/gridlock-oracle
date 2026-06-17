"""
find_high_risk_example.py — find REAL events that the model scores as HIGH closure
risk, and print every field so you can reproduce the high prediction in the UI.

Run: python find_high_risk_example.py --data data/flipkart_gridlock.csv
"""

import argparse
import pandas as pd
from predictor import GridlockPredictor

UI_FIELDS = ["event_cause", "veh_type", "junction", "zone", "corridor",
             "police_station", "priority", "event_type"]


def main(path, n=5):
    df = pd.read_csv(path)
    predictor = GridlockPredictor()

    # only consider events that actually required a closure (so UI test is meaningful)
    mask = df["requires_road_closure"].astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])
    pos = df[mask].copy()

    sample = pos.sample(min(400, len(pos)), random_state=0)
    scored = []
    for _, row in sample.iterrows():
        e = {f: row[f] for f in UI_FIELDS + ["start_datetime"]
             if f in row and pd.notna(row[f])}
        prob = predictor.predict(e)["closure_prob"]
        scored.append((prob, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop {n} real CLOSURE events the model scores highest — enter these EXACT")
    print("values in the Predict Event page to see a high probability:\n")
    for i, (prob, row) in enumerate(scored[:n], 1):
        print(f"════════ Example {i} — predicted closure prob: {prob:.1%} ════════")
        for f in UI_FIELDS:
            val = row[f] if f in row and pd.notna(row[f]) else "(any)"
            print(f"   {f:16s}: {val}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)