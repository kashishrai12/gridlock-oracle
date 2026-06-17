"""
try_real_event.py — run the predictor on ACTUAL rows from the dataset.

Picks a few real events (including one that actually required a closure and one that
didn't), feeds their real attributes to GridlockPredictor, and prints the prediction
next to the ground truth so you can sanity-check.

Run: python try_real_event.py --data data/flipkart_gridlock.csv
"""

import argparse, json
import pandas as pd
from predictor import GridlockPredictor

FIELDS = ["start_datetime", "priority", "event_type", "event_cause",
          "veh_type", "junction", "corridor", "zone", "police_station"]


def to_event(row):
    e = {}
    for f in FIELDS:
        if f in row and pd.notna(row[f]):
            e[f] = row[f]
    return e


def show(predictor, row, tag):
    actual = row.get("requires_road_closure")
    actual_str = str(actual)
    e = to_event(row)
    r = predictor.predict(e)
    print(f"\n──────── {tag} ────────")
    print(f"  cause={e.get('event_cause')!r}  veh={e.get('veh_type')!r}  "
          f"junction={e.get('junction')!r}  priority={e.get('priority')!r}")
    print(f"  ACTUAL closure required : {actual_str}")
    print(f"  PREDICTED closure prob  : {r['closure_prob']:.1%}  "
          f"-> {'FLAG (barricade)' if r['resources']['barricading_recommended'] else 'no barricade'}")
    print(f"  impact {r['impact_score']}/10 ({r['impact_tier']})  "
          f"| expected clearance ~{r['expected_clearance_mins']} min  "
          f"| known location: {r['is_known_location']}")
    print(f"  top reasons:")
    for ex in r["explanations"][:3]:
        print(f"     {ex['direction']:6s} {ex['feature']:22s} ({ex['contribution']:+.2f})")


def main(path):
    df = pd.read_csv(path)
    predictor = GridlockPredictor()

    def closure_mask(v):
        return df["requires_road_closure"].astype(str).str.lower().isin(["1", "true", "yes", "y", "t"]) == v

    # one real event that DID need a closure, one that did NOT
    pos = df[closure_mask(True)]
    neg = df[closure_mask(False)]
    if not pos.empty:
        show(predictor, pos.sample(1, random_state=1).iloc[0].to_dict(), "REAL EVENT — actually needed closure")
    if not neg.empty:
        show(predictor, neg.sample(1, random_state=1).iloc[0].to_dict(), "REAL EVENT — no closure needed")

    # plus 3 random real events
    for i, (_, row) in enumerate(df.sample(min(3, len(df)), random_state=7).iterrows(), 1):
        show(predictor, row.to_dict(), f"RANDOM REAL EVENT #{i}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)