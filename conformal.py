"""
conformal.py — CONFORMAL PREDICTION for the closure classifier.

A single probability ("35%") hides how sure the model is. Conformal prediction wraps the
model to output a prediction SET with a statistical coverage guarantee: at 90% confidence,
the true label is in the set >=90% of the time. For our binary closure problem each event
becomes one of:

  • {needs closure}     -> confident YES
  • {no closure}        -> confident NO
  • {both} (UNCERTAIN)  -> the model abstains; send to a human

This is honest uncertainty quantification — a named gap in the traffic-ML literature — and
operationally it routes only the genuinely ambiguous incidents to a person.

Method: split-conformal (Vovk). Nonconformity score = 1 - P(true class) on a calibration
split; the (1-alpha) quantile sets the inclusion threshold. Self-contained.

Run: python conformal.py --data data/flipkart_gridlock.csv
Outputs: models/conformal.pkl  (the two thresholds + measured coverage)
"""

import argparse, pickle
import numpy as np
import xgboost as xgb
from utils import preprocess as pp

MODELS_DIR = "models"
ALPHA = 0.10          # target 90% coverage


def main(path, alpha=ALPHA):
    df = pp.load_and_prepare(path)
    _, test_df = pp.temporal_split(df, 0.8)
    stats = pickle.load(open(f"{MODELS_DIR}/location_stats.pkl", "rb"))
    feats, cats = pp.classifier_feature_cols(stats)
    _, _, test_df = pp.build_xy(test_df, stats)
    X = test_df[feats].copy()
    for c in feats:
        if c not in cats:
            X[c] = X[c].astype(float)
    y = test_df["closure_int"].astype(int).values

    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model(f"{MODELS_DIR}/closure_clf.json")
    p1 = clf.predict_proba(X)[:, 1]
    # use the SAME calibrated probability the predictor/dashboard displays, so the conformal
    # decision is consistent with the shown percentage
    
    P = np.column_stack([1 - p1, p1])          # P[:,0]=no, P[:,1]=yes

    # split test into calibration / evaluation
    n = len(y); cut = n // 2
    cal_idx, ev_idx = np.arange(cut), np.arange(cut, n)
    # nonconformity on calibration = 1 - prob of TRUE class
    scores = 1 - P[cal_idx, y[cal_idx]]
    qhat = np.quantile(scores, np.ceil((len(cal_idx) + 1) * (1 - alpha)) / len(cal_idx))

    # build prediction sets on eval: include class k if 1 - P[k] <= qhat
    include0 = (1 - P[ev_idx, 0]) <= qhat
    include1 = (1 - P[ev_idx, 1]) <= qhat
    set_size = include0.astype(int) + include1.astype(int)
    ytrue = y[ev_idx]
    covered = ((ytrue == 0) & include0) | ((ytrue == 1) & include1)

    coverage = float(covered.mean())
    uncertain = float((set_size == 2).mean())
    confident = float((set_size == 1).mean())
    empty = float((set_size == 0).mean())

    pickle.dump({"qhat": float(qhat), "alpha": alpha, "coverage": coverage,
                 "uncertain_rate": uncertain, "confident_rate": confident},
                open(f"{MODELS_DIR}/conformal.pkl", "wb"))

    print(f"\n===== CONFORMAL PREDICTION (target {int(100*(1-alpha))}% coverage) =====")
    print(f"  empirical coverage: {coverage:.1%}  (guarantee: >= {int(100*(1-alpha))}%)")
    print(f"  confident calls (single-label set): {confident:.1%}")
    print(f"  UNCERTAIN (abstains, route to human):  {uncertain:.1%}")
    if empty:
        print(f"  empty sets (very confident region): {empty:.1%}")
    print(f"\n  Operationally: the model commits on {confident:.0%} of incidents and flags "
          f"{uncertain:.0%} as too ambiguous to auto-decide.")
    print(f"\n[saved] models/conformal.pkl\n")


def classify(p_closure, qhat):
    """Turn a closure probability into a conformal decision label using saved qhat."""
    inc0 = (1 - (1 - p_closure)) <= qhat      # include 'no'  if 1-P0 <= qhat
    inc1 = (1 - p_closure) <= qhat            # include 'yes' if 1-P1 <= qhat
    if inc1 and not inc0:
        return "CONFIDENT: needs closure"
    if inc0 and not inc1:
        return "CONFIDENT: no closure"
    if inc0 and inc1:
        return "UNCERTAIN: review"
    return "EDGE"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args()
    main(a.data, a.alpha)