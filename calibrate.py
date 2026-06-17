"""
calibrate.py — Probability calibration for the closure classifier.

scale_pos_weight (used to handle the 10.5% imbalance) inflates the raw probabilities,
so they rank well but don't mean what they say. Calibration learns a monotonic map from
raw score -> true frequency, so a calibrated "30%" really means ~30% of such events need
a closure. We fit on a held-out slice and evaluate on a separate slice (no leakage),
report Brier score + Expected Calibration Error (ECE) before/after, and keep whichever
is best — falling back to raw if calibration doesn't help.

Run: python calibrate.py --data data/flipkart_gridlock.csv
Outputs: models/closure_calibrator.pkl, models/calibration_curve.csv, models/calibration_metrics.json
"""

import argparse, json, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import xgboost as xgb

from utils import preprocess as pp

MODELS_DIR = "models"


def reliability(y, p, bins=10):
    """Reliability-diagram bins + Expected Calibration Error."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece, n = [], 0.0, len(y)
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf, acc, w = p[m].mean(), y[m].mean(), m.sum() / n
        ece += w * abs(conf - acc)
        rows.append({"mean_pred": round(float(conf), 3), "frac_pos": round(float(acc), 3),
                     "count": int(m.sum())})
    return pd.DataFrame(rows), float(ece)


def main(path):
    # same temporal split + SAME saved stats as the deployed model (no refit, no leakage)
    df = pp.load_and_prepare(path)
    _, test_df = pp.temporal_split(df, 0.8)
    with open(f"{MODELS_DIR}/location_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    feats, cats = pp.classifier_feature_cols(stats)
    _, _, test_df = pp.build_xy(test_df, stats)

    Xte = test_df[feats].copy()
    for c in feats:
        if c not in cats:
            Xte[c] = Xte[c].astype(float)
    yte = test_df["closure_int"].astype(int).values

    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model(f"{MODELS_DIR}/closure_clf.json")
    raw = clf.predict_proba(Xte)[:, 1]

    # split the held-out test set in time: first half calibrates, second half evaluates
    m = len(yte); cut = m // 2
    raw_cal, y_cal = raw[:cut], yte[:cut]
    raw_ev, y_ev = raw[cut:], yte[cut:]

    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)
    platt = LogisticRegression().fit(raw_cal.reshape(-1, 1), y_cal)
    iso_ev = iso.predict(raw_ev)
    platt_ev = platt.predict_proba(raw_ev.reshape(-1, 1))[:, 1]

    res = {}
    for name, pv in [("raw", raw_ev), ("isotonic", iso_ev), ("platt", platt_ev)]:
        _, ece = reliability(y_ev, pv)
        res[name] = {"brier": round(float(brier_score_loss(y_ev, pv)), 4), "ece": round(ece, 4)}

    best = min(["isotonic", "platt"], key=lambda k: res[k]["brier"])
    chosen = best if res[best]["brier"] <= res["raw"]["brier"] else "raw"

    calibrator = {"method": chosen,
                  "model": (iso if chosen == "isotonic" else platt if chosen == "platt" else None)}
    with open(f"{MODELS_DIR}/closure_calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

    chosen_pv = {"raw": raw_ev, "isotonic": iso_ev, "platt": platt_ev}[chosen]
    rc_raw, _ = reliability(y_ev, raw_ev); rc_raw["which"] = "raw (uncalibrated)"
    rc_cal, _ = reliability(y_ev, chosen_pv); rc_cal["which"] = f"calibrated ({chosen})"
    pd.concat([rc_raw, rc_cal]).to_csv(f"{MODELS_DIR}/calibration_curve.csv", index=False)
    json.dump({"metrics": res, "chosen": chosen, "n_calib": int(cut), "n_eval": int(m - cut)},
              open(f"{MODELS_DIR}/calibration_metrics.json", "w"), indent=2)

    print("calibration (evaluated on held-out half of the temporal test set):")
    print(f"  {'method':<10}{'Brier':>9}{'ECE':>9}   (lower = better)")
    for k, v in res.items():
        print(f"  {k:<10}{v['brier']:>9}{v['ece']:>9}")
    print(f"  -> chosen: {chosen}")
    if chosen != "raw":
        imp = round(100 * (res['raw']['ece'] - res[chosen]['ece']) / max(res['raw']['ece'], 1e-6), 0)
        print(f"  -> calibration cut ECE by ~{imp:.0f}%")
    print(f"[done] saved closure_calibrator.pkl, calibration_curve.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)