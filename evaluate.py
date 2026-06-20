
import argparse, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from utils import preprocess as pp
from analogs import AnalogRetriever

MODELS_DIR = "models"
OPERATING_THRESHOLD = 0.25


def _confusion(y, flagged):
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    tn = int(((y == 0) & ~flagged).sum())
    return tp, fp, fn, tn


def main(path):
    # rebuild the exact held-out test set + deployed model
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
    y = test_df["closure_int"].astype(int).values

    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model(f"{MODELS_DIR}/closure_clf.json")
    p = clf.predict_proba(Xte)[:, 1]            # RAW score = the operating/ranking score
    # Calibrated probabilities are kept for DISPLAY only (they make the % trustworthy), but
    # the barricade DECISION ranks on the raw score, which preserves the recall/precision
    # trade-off that calibration compresses. Standard practice: rank raw, calibrate to show.
    p_cal = p.copy()
    try:
        with open(f"{MODELS_DIR}/closure_calibrator.pkl", "rb") as f:
            cal = pickle.load(f)
        if cal.get("method") == "isotonic":
            p_cal = np.clip(cal["model"].predict(p), 0, 1)
        elif cal.get("method") == "platt":
            p_cal = cal["model"].predict_proba(p.reshape(-1, 1))[:, 1]
    except Exception:
        pass

    n = len(y); closures = int(y.sum())
    # per-event officer-hours (realised): personnel from policy x actual clearance hours
    clr = pd.to_numeric(test_df["clearance_mins"], errors="coerce").fillna(60).clip(upper=480).values
    personnel = np.array([AnalogRetriever.resource_plan(int(c), float(pi))["personnel"]
                          for c, pi in zip(clr, p)])
    hours = clr / 60.0
    event_officer_hours = personnel * hours
    blanket_hours = float(event_officer_hours.sum())

    # ---- threshold sweep ----
    sweep = []
    for t in [0.15, 0.20, 0.25, 0.30, 0.35]:
        flagged = p >= t
        tp, fp, fn, tn = _confusion(y, flagged)
        recall = tp / closures if closures else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        targeted_hours = float(event_officer_hours[flagged].sum())
        sweep.append({
            "threshold": t,
            "flagged_pct": round(100 * flagged.mean(), 1),
            "closures_covered_pct": round(100 * recall, 1),
            "precision": round(precision, 3),
            "false_alarms": fp,
            "missed_closures": fn,
            "officer_hours_used": round(targeted_hours),
            "officer_hours_vs_blanket_pct": round(100 * targeted_hours / blanket_hours, 1) if blanket_hours else 0,
        })
    pd.DataFrame(sweep).to_csv(f"{MODELS_DIR}/backtest_threshold_curve.csv", index=False)

    # ---- operating point ----
    flagged = p >= OPERATING_THRESHOLD
    tp, fp, fn, tn = _confusion(y, flagged)
    recall = tp / closures if closures else 0
    targeted_hours = float(event_officer_hours[flagged].sum())
    non_closures = n - closures
    wasted_reduction = (non_closures - fp) / non_closures if non_closures else 0

    # ---- tiered readiness coverage (graded posture, not binary) ----
    import sys as _sys
    from utils import preprocess as _pp
    tiers = np.array([_pp.readiness_tier(pi) for pi in p])
    cl = (y == 1)
    pre = (tiers == "PRE-POSITION")
    sby = (tiers == "STANDBY")
    mon = (tiers == "MONITOR")
    closures_pre = int((cl & pre).sum())
    closures_sby = int((cl & sby).sum())
    closures_mon = int((cl & mon).sum())
    any_readiness = closures_pre + closures_sby
    tiered = {
        "preposition_closures_pct": round(100 * closures_pre / closures, 1) if closures else 0,
        "standby_closures_pct": round(100 * closures_sby / closures, 1) if closures else 0,
        "monitor_missed_closures_pct": round(100 * closures_mon / closures, 1) if closures else 0,
        "any_readiness_recall_pct": round(100 * any_readiness / closures, 1) if closures else 0,
        "events_preposition_pct": round(100 * pre.mean(), 1),
        "events_standby_pct": round(100 * sby.mean(), 1),
    }

    report = {
        "test_events": n,
        "actual_closures": closures,
        "operating_threshold": OPERATING_THRESHOLD,
        "system": {
            "closures_pre_positioned": tp,
            "closures_covered_pct": round(100 * recall, 1),
            "events_flagged_pct": round(100 * flagged.mean(), 1),
            "false_alarms": fp,
            "missed_closures": fn,
            "officer_hours": round(targeted_hours),
        },
        "blanket": {
            "closures_covered_pct": 100.0,
            "events_flagged_pct": 100.0,
            "false_alarms": non_closures,
            "officer_hours": round(blanket_hours),
        },
        "reactive": {"closures_covered_pct": 0.0, "events_flagged_pct": 0.0, "officer_hours": 0},
        "tiered_readiness": tiered,
        "impact": {
            "wasted_deployments_avoided_pct": round(100 * wasted_reduction, 1),
            "officer_hours_saved_vs_blanket": round(blanket_hours - targeted_hours),
            "officer_hours_saved_pct": round(100 * (1 - targeted_hours / blanket_hours), 1) if blanket_hours else 0,
            "efficiency_multiple": round((recall) / (flagged.mean() or 1), 1),
        },
    }
    with open(f"{MODELS_DIR}/backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- print ----
    print(f"\n===== BACKTEST on held-out test set ({n} events, {closures} real closures) =====\n")
    print(f"{'threshold':>9}{'flagged%':>10}{'closures cov%':>14}{'precision':>11}{'false alarms':>14}{'missed':>8}")
    for s in sweep:
        print(f"{s['threshold']:>9}{s['flagged_pct']:>10}{s['closures_covered_pct']:>14}"
              f"{s['precision']:>11}{s['false_alarms']:>14}{s['missed_closures']:>8}")

    op = report["system"]; im = report["impact"]
    print(f"\n--- At operating threshold {OPERATING_THRESHOLD} ---")
    print(f"  Pre-positioned barricades for {op['closures_covered_pct']}% of real closures "
          f"({op['closures_pre_positioned']}/{closures})")
    print(f"  while flagging only {op['events_flagged_pct']}% of events "
          f"({im['efficiency_multiple']}x more efficient than blanket).")
    print(f"  Wasted deployments avoided vs blanket: {im['wasted_deployments_avoided_pct']}%")
    print(f"  Estimated officer-hours saved vs blanket: {im['officer_hours_saved_vs_blanket']:,} "
          f"({im['officer_hours_saved_pct']}%)")

    tr = report["tiered_readiness"]
    print(f"\n--- Tiered readiness (graded posture, not binary) ---")
    print(f"  Of real closures:  PRE-POSITION {tr['preposition_closures_pct']}%  |  "
          f"STANDBY {tr['standby_closures_pct']}%  |  unwatched {tr['monitor_missed_closures_pct']}%")
    print(f"  ANY-readiness recall (pre-position OR standby): {tr['any_readiness_recall_pct']}% of closures")
    print(f"  Resource posture: full deploy on {tr['events_preposition_pct']}% of events, "
          f"staged on {tr['events_standby_pct']}%")
    print(f"\n[done] saved backtest_report.json, backtest_threshold_curve.csv\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)