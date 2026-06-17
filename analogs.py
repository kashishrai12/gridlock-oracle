"""
analogs.py — Historical analog retrieval (k-nearest-neighbour).

The dataset records what HAPPENED to past events (how long they took to clear, whether
they needed a closure) but NOT how many officers/barricades were deployed. So instead of
inventing staffing numbers, we:

  1. Find the K most SIMILAR past incidents (same kind of event, vehicle, area, time).
  2. Report their real outcomes: clearance-time distribution + closure rate.
  3. Derive a resource plan from those data-grounded quantities via a transparent,
     tunable staffing policy (e.g. "1 officer per 30 min of expected clearance").
  4. Surface the actual analog incidents so the recommendation is auditable —
     "we're not guessing; here are the closest historical precedents."

Build:  python analogs.py --data models/enriched_dataset.csv
Use:    from analogs import AnalogRetriever; AnalogRetriever().query(event_dict)
"""

import argparse, os, pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

from utils import preprocess as pp

def _clean(v):
    s = str(v).strip()
    return "—" if s.lower() in ("", "nan", "none") else s

MODELS_DIR = "models"
INDEX_PATH = f"{MODELS_DIR}/analog_index.pkl"

# situational features used for similarity (NOT outcomes like closure/clearance)
CAT_FEATURES = ["event_cause", "veh_type"]
NUM_FEATURES = ["priority_encoded", "is_planned", "hour", "day_of_week"]
K_DEFAULT = 15

# staffing policy — tunable, stated openly. Anchored to data-grounded quantities.
OFFICERS_PER_30MIN = 1.0       # +1 officer per 30 min of expected clearance
OFFICERS_BASE = 2.0
OFFICERS_PER_CLOSURE_PROB = 4.0  # closure risk adds up to 4 officers


def _engineer(df):
    """Reuse the training feature path so query/index features match exactly."""
    df = pp._parse_datetimes(df)
    if "start_datetime" in df.columns:
        df = pp._time_features(df)
    df = pp._static_features(df)
    return df


def build_index(data_path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(data_path)
    df = df[df["clearance_mins"].notna()].copy()
    for c in CAT_FEATURES:
        if c not in df.columns:
            df[c] = "UNK"
        df[c] = df[c].astype(str).str.strip().replace({"nan": "none", "": "none", "None": "none"})
    for c in NUM_FEATURES:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])
    X = pre.fit_transform(df[CAT_FEATURES + NUM_FEATURES])
    if hasattr(X, "toarray"):
        X = X.toarray()
    nn = NearestNeighbors(n_neighbors=min(K_DEFAULT, len(df)), metric="euclidean")
    nn.fit(X)

    # reference table with the outcomes + readable fields
    keep = CAT_FEATURES + ["clearance_mins", "closure_int"]
    for extra in ["junction", "priority", "event_cause"]:
        if extra in df.columns and extra not in keep:
            keep.append(extra)
    ref = df[keep].reset_index(drop=True)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"pre": pre, "nn": nn, "ref": ref,
                     "cat": CAT_FEATURES, "num": NUM_FEATURES}, f)
    print(f"[analogs] indexed {len(df)} historical events; k={nn.n_neighbors}")
    print(f"[analogs] median clearance overall: {df['clearance_mins'].median():.0f} min")
    print(f"[analogs] saved {INDEX_PATH}")
    return ref


class AnalogRetriever:
    def __init__(self, models_dir=MODELS_DIR):
        with open(f"{models_dir}/analog_index.pkl", "rb") as f:
            d = pickle.load(f)
        self.pre, self.nn, self.ref = d["pre"], d["nn"], d["ref"]
        self.cat, self.num = d["cat"], d["num"]

    def query(self, event: dict, k=K_DEFAULT, closure_prob=None):
        df = pd.DataFrame([dict(event)])
        df = _engineer(df)
        for c in self.cat:
            df[c] = (df[c].astype(str).str.strip().replace({"nan": "none", "": "none", "None": "none"})
                     if c in df.columns else "none")
        for c in self.num:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0

        X = self.pre.transform(df[self.cat + self.num])
        if hasattr(X, "toarray"):
            X = X.toarray()
        k = min(k, self.nn.n_neighbors)
        dist, idx = self.nn.kneighbors(X, n_neighbors=k)
        rows = self.ref.iloc[idx[0]].copy()

        clearances = rows["clearance_mins"].values
        result = {
            "n_matched": int(len(rows)),
            "expected_clearance_mins": int(np.median(clearances)),
            "clearance_p25": int(np.percentile(clearances, 25)),
            "clearance_p75": int(np.percentile(clearances, 75)),
            "analog_closure_rate": round(float(rows["closure_int"].mean()), 3),
            "analogs": [
                {"event_cause": _clean(r.get("event_cause", "")),
                 "veh_type": _clean(r.get("veh_type", "")),
                 "zone": _clean(r.get("zone", "")),
                 "clearance_mins": int(r["clearance_mins"]),
                 "needed_closure": bool(r["closure_int"])}
                for _, r in rows.head(8).iterrows()
            ],
        }
        if closure_prob is not None:
            result["resources"] = self.resource_plan(result["expected_clearance_mins"], closure_prob)
            result["resources"]["basis"] = f"median of {len(rows)} similar historical incidents"
        return result

    @staticmethod
    def resource_plan(expected_clearance_mins, closure_prob):
        """Transparent staffing policy anchored to DATA-GROUNDED expected clearance and
        the model's closure probability. Coefficients are policy (stated), not invented
        per-event guesses — and every term traces to a real quantity."""
        officers = (OFFICERS_BASE
                    + OFFICERS_PER_30MIN * (expected_clearance_mins / 30.0)
                    + OFFICERS_PER_CLOSURE_PROB * closure_prob)
        officers = int(np.clip(round(officers), 2, 20))
        barricades = (4 if closure_prob >= 0.6 else 2 if closure_prob >= 0.3 else 0)
        supervisors = 1 + int(officers >= 10)
        return {
            "personnel": officers,
            "supervisors": supervisors,
            "barricades": barricades,
            "barricading_recommended": closure_prob >= 0.3,
            "diversion_recommended": bool(closure_prob >= 0.5 or expected_clearance_mins >= 120),
            "rapid_response_required": bool(closure_prob >= 0.6 and expected_clearance_mins >= 90),
            "expected_clearance_mins": int(expected_clearance_mins),
            "basis": "median of {} similar historical incidents".format("K"),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="models/enriched_dataset.csv")
    build_index(ap.parse_args().data)
    # smoke demo
    try:
        r = AnalogRetriever().query(
            {"start_datetime": "2024-06-01 18:00", "priority": "High",
             "event_type": "unplanned", "event_cause": "Accident",
             "veh_type": "Truck", "zone": "Z_1"}, closure_prob=0.5)
        import json
        print("\n[demo] analog query:")
        print(json.dumps({k: v for k, v in r.items() if k != "analogs"}, indent=2))
        print("top analogs:", r["analogs"][:3])
    except Exception as e:
        print(f"[demo] skipped ({e})")