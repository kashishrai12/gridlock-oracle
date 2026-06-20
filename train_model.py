
import argparse, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
import xgboost as xgb

from utils import preprocess as pp

MODELS_DIR = "models"
TARGET = "closure_int"                      # 1 = event required a road closure
# closure-derived aggregates are excluded to keep the classifier leakage-clean
EXCLUDE = {"closure_int", "junction_closure_rate", "corridor_closure_rate"}


def classifier_features(stats):
    all_cols, cat_cols = pp.get_feature_cols(stats)
    feats = [c for c in all_cols if c not in EXCLUDE]
    cats = [c for c in cat_cols if c not in EXCLUDE]
    return feats, cats


def operating_points(y_true, proba):
    """Precision/recall at thresholds relevant to a barricading decision (favour recall:
    missing an event that needs closure is worse than a false alarm)."""
    rows = []
    for t in [0.20, 0.30, 0.50]:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y_true, pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, pred, zero_division=0), 3),
            "flagged_pct": round(100 * pred.mean(), 1),
        })
    return rows


def new_impact_score(p_closure, priority_encoded, is_planned, junction_hist_clearance, global_clear):
    """Transparent 0-10 impact score, now anchored on the VALIDATED closure probability
    plus context. No circular label: P(closure) comes from the model, the rest is policy."""
    s  = 5.5 * p_closure                                   # 0..5.5  (dominant, validated term)
    s += 1.5 * (priority_encoded / 3.0)                    # 0..1.5
    s += 1.0 * is_planned                                  # planned events get lead-time weight
    s += 2.0 * np.tanh(junction_hist_clearance / max(global_clear, 1.0) - 1.0 + 1.0) \
         if junction_hist_clearance else 0.0               # 0..~2 location history
    return round(float(np.clip(s, 0, 10)), 1)


def main(data_path):
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pp.load_and_prepare(data_path)
    print(f"[data] {len(df)} events with clean target/features")

    train_df, test_df = pp.temporal_split(df, train_frac=0.8)
    print(f"[split] train={len(train_df)}  test={len(test_df)} (temporal)")

    stats = pp.fit_location_stats(train_df)
    _, _, train_df = pp.build_xy(train_df, stats)
    _, _, test_df = pp.build_xy(test_df, stats)

    feats, cats = pp.classifier_feature_cols(stats)
    Xtr, ytr = train_df[feats].copy(), train_df[TARGET].astype(int)
    Xte, yte = test_df[feats].copy(), test_df[TARGET].astype(int)
    for c in feats:                                   # ensure dtypes
        if c not in cats:
            Xtr[c] = Xtr[c].astype(float); Xte[c] = Xte[c].astype(float)

    pos_rate = ytr.mean()
    spw = (1 - pos_rate) / max(pos_rate, 1e-6)        # handle 10% imbalance
    clf = xgb.XGBClassifier(
        n_estimators=450, max_depth=5, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        scale_pos_weight=spw, tree_method="hist", enable_categorical=True,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    ap = average_precision_score(yte, proba)
    base_ap = yte.mean()                              # PR baseline = positive rate

    print(f"\n=== CLOSURE-NEED CLASSIFIER (test, temporal) ===")
    print(f"  positive rate (test): {yte.mean():.1%}")
    print(f"  ROC-AUC : {auc:.3f}   (0.5 = useless)")
    print(f"  PR-AUC  : {ap:.3f}   (baseline {base_ap:.3f}, lift {ap/base_ap:.1f}x)")
    print(f"  operating points (barricading decision):")
    ops = operating_points(yte.values, proba)
    print(f"    {'thr':>4} {'prec':>6} {'recall':>7} {'f1':>6} {'flagged%':>9}")
    for o in ops:
        print(f"    {o['threshold']:>4} {o['precision']:>6} {o['recall']:>7} {o['f1']:>6} {o['flagged_pct']:>9}")

    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("\n=== TOP FEATURE IMPORTANCE (gain) ===")
    for name, val in imp.head(10).items():
        print(f"  {name:26s} {val:.3f}")

    # ---- persist (JSON model => categorical-safe + SHAP-capable) ----
    clf.save_model(f"{MODELS_DIR}/closure_clf.json")
    with open(f"{MODELS_DIR}/location_stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    if stats["junction"] is not None:
        stats["junction"].reset_index().to_csv(f"{MODELS_DIR}/junction_risk.csv", index=False)
    if stats["corridor"] is not None:
        stats["corridor"].reset_index().to_csv(f"{MODELS_DIR}/corridor_stats.csv", index=False)
    if stats["zone"] is not None:
        stats["zone"].reset_index().to_csv(f"{MODELS_DIR}/zone_stats.csv", index=False)

    # enriched dataset: closure probability + new impact score (+ descriptive clearance)
    _, _, full = pp.build_xy(df.copy(), stats)
    Xfull = full[feats].copy()
    for c in feats:
        if c not in cats:
            Xfull[c] = Xfull[c].astype(float)
    full["pred_closure_prob"] = clf.predict_proba(Xfull)[:, 1]
    g_clear = stats["global"]["clear"]
    full["impact_score"] = [
        pp.closure_impact_score(p, pe, ip, jh, g_clear)
        for p, pe, ip, jh in zip(full["pred_closure_prob"], full["priority_encoded"],
                                 full["is_planned"], full["junction_hist_clearance"])
    ]
    full["impact_tier"] = full["impact_score"].map(pp.impact_tier)
    full.to_csv(f"{MODELS_DIR}/enriched_dataset.csv", index=False)

    metrics = {"roc_auc": round(float(auc), 3), "pr_auc": round(float(ap), 3),
               "pr_baseline": round(float(base_ap), 3), "operating_points": ops,
               "n_train": len(train_df), "n_test": len(test_df),
               "pos_rate_train": round(float(pos_rate), 3), "features": feats}
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] artifacts written to {MODELS_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)