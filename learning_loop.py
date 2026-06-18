"""
learning_loop.py — a REAL feedback-and-recalibration loop.

Two parts:
  1. simulate(): replays events chronologically in batches. After each batch its true
     outcomes are added to the feedback pool and the probability calibrator is refit, so
     later batches are scored by a calibrator trained on everything seen so far. The
     resulting curve shows calibration error (ECE) dropping as the system accumulates
     feedback — i.e. it demonstrably gets better with use.
  2. Deployable hooks: log_outcome() records each (prediction, actual) as it resolves;
     recalibrate_from_feedback() refits the live calibrator from that log. This is the
     mechanism a real deployment would run nightly.

Run: python learning_loop.py --data data/flipkart_gridlock.csv
"""

import argparse, csv, os, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import xgboost as xgb

from utils import preprocess as pp

MODELS_DIR = "models"
FEEDBACK = f"{MODELS_DIR}/feedback_log.csv"
CURVE = f"{MODELS_DIR}/learning_curve.csv"


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e, n = 0.0, len(y)
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def _raw_probs_chrono(data_path):
    df = pp.load_and_prepare(data_path)
    with open(f"{MODELS_DIR}/location_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    feats, cats = pp.classifier_feature_cols(stats)
    _, _, df = pp.build_xy(df, stats)
    X = df[feats].copy()
    for c in feats:
        if c not in cats:
            X[c] = X[c].astype(float)
    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model(f"{MODELS_DIR}/closure_clf.json")
    p = clf.predict_proba(X)[:, 1]
    y = df["closure_int"].astype(int).to_numpy()
    ts = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).to_numpy()
    order = np.argsort(ts)
    return p[order], y[order]


def simulate(data_path, batches=10):
    p, y = _raw_probs_chrono(data_path)
    n = len(p); bs = max(1, n // batches)
    rows, acc_p, acc_y, cal = [], [], [], None

    for i in range(batches):
        s = i * bs
        e = n if i == batches - 1 else (i + 1) * bs
        bp, by = p[s:e], y[s:e]
        if len(bp) == 0:
            continue
        cp = cal.predict(bp) if cal is not None else bp     # score with feedback so far
        try:
            brier = brier_score_loss(by, cp) if len(set(by)) > 1 else np.nan
        except Exception:
            brier = np.nan
        rows.append({"batch": i + 1, "cum_events": e,
                     "ece_uncalibrated": round(_ece(by, bp), 4),
                     "ece_calibrated": round(_ece(by, cp), 4),
                     "brier": round(float(brier), 4) if brier == brier else None})
        # accumulate outcomes, refit calibrator (the "learning" step)
        acc_p.extend(bp.tolist()); acc_y.extend(by.tolist())
        if len(set(acc_y)) > 1:
            cal = IsotonicRegression(out_of_bounds="clip").fit(np.array(acc_p), np.array(acc_y))

    out = pd.DataFrame(rows)
    out.to_csv(CURVE, index=False)
    # robust headline: average across post-cold-start batches (batch 1 has no feedback yet)
    warm = out.iloc[1:] if len(out) > 1 else out
    mean_uncal = warm["ece_uncalibrated"].mean()
    mean_cal = warm["ece_calibrated"].mean()
    print(f"[learning] replayed {n} events in {len(out)} feedback batches -> {CURVE}")
    print(f"[learning] mean ECE once feedback is flowing: {mean_uncal:.3f} uncalibrated "
          f"-> {mean_cal:.3f} with continuous relearning")
    if mean_uncal > 0:
        print(f"[learning] {100*(mean_uncal-mean_cal)/mean_uncal:.0f}% lower calibration error "
              f"by learning from outcomes")
    print(out.to_string(index=False))
    return out


# ----------------------------- deployable hooks ----------------------------- #
def log_outcome(event_id, predicted, actual, path=FEEDBACK):
    """Call when an incident resolves: stores prediction vs reality for nightly relearning."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["event_id", "predicted", "actual", "logged_at"])
        w.writerow([event_id, round(float(predicted), 4), int(actual),
                    pd.Timestamp.utcnow().isoformat()])


def recalibrate_from_feedback(path=FEEDBACK, min_n=50):
    """Refit the live calibrator from logged outcomes. Returns n used, or None."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if len(d) < min_n or d["actual"].nunique() < 2:
        return None
    cal = IsotonicRegression(out_of_bounds="clip").fit(d["predicted"].values, d["actual"].values)
    with open(f"{MODELS_DIR}/closure_calibrator.pkl", "wb") as f:
        pickle.dump({"method": "isotonic", "model": cal}, f)
    return len(d)


def feedback_stats(path=FEEDBACK):
    if not os.path.exists(path):
        return {"n": 0}
    d = pd.read_csv(path)
    return {"n": len(d), "closures": int(d["actual"].sum()),
            "avg_predicted": round(float(d["predicted"].mean()), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batches", type=int, default=10)
    a = ap.parse_args()
    simulate(a.data, a.batches)