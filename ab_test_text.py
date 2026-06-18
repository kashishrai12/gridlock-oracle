"""
ab_test_text.py — honest A/B: does the `description` text actually improve the closure
model? Trains the SAME XGBoost classifier (identical config to train_model.py) twice on
the temporal split — once on structured features only, once with the keyword text
features added — and reports ROC-AUC + PR-AUC on the held-out test set, plus which text
features the model found useful.

Decision rule: integrate only if PR-AUC improves meaningfully. If not, keep the keywords
as a display-only signal (still useful, no false claim of lift).

Run: python ab_test_text.py --data data/flipkart_gridlock.csv
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import preprocess as pp
from text_features import add_text_features, TEXT_FEATURES


def fit_eval(Xtr, ytr, Xte, yte):
    spw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
    clf = xgb.XGBClassifier(
        n_estimators=450, max_depth=5, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        scale_pos_weight=spw, tree_method="hist", enable_categorical=True,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr, ytr, verbose=False)
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p), clf


def main(path):
    df = pp.load_and_prepare(path)
    train_df, test_df = pp.temporal_split(df, 0.8)
    stats = pp.fit_location_stats(train_df)
    _, _, train_df = pp.build_xy(train_df, stats)
    _, _, test_df = pp.build_xy(test_df, stats)

    feats, cats = pp.classifier_feature_cols(stats)
    ytr = train_df["closure_int"].astype(int)
    yte = test_df["closure_int"].astype(int)
    Xtr, Xte = train_df[feats].copy(), test_df[feats].copy()
    for c in feats:
        if c not in cats:
            Xtr[c] = Xtr[c].astype(float); Xte[c] = Xte[c].astype(float)

    # ---- baseline: structured features only ----
    auc0, ap0, _ = fit_eval(Xtr, ytr, Xte, yte)

    # ---- + text features ----
    ttr = add_text_features(train_df)
    tte = add_text_features(test_df)
    Xtr2 = pd.concat([Xtr, ttr], axis=1)
    Xte2 = pd.concat([Xte, tte], axis=1)
    auc1, ap1, clf1 = fit_eval(Xtr2, ytr, Xte2, yte)

    base = yte.mean()
    print(f"\n===== A/B: does `description` text help? (test set, {len(yte)} events, "
          f"{int(yte.sum())} closures, base rate {base:.1%}) =====\n")
    print(f"{'variant':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'PR lift vs base':>17}")
    print(f"{'structured only':<26}{auc0:>9.3f}{ap0:>9.3f}{ap0/base:>15.2f}x")
    print(f"{'structured + text':<26}{auc1:>9.3f}{ap1:>9.3f}{ap1/base:>15.2f}x")
    d_auc, d_ap = auc1 - auc0, ap1 - ap0
    print(f"\n  delta ROC-AUC: {d_auc:+.3f}   delta PR-AUC: {d_ap:+.3f}")

    # how much did the text features actually get used?
    imp = pd.Series(clf1.feature_importances_, index=Xtr2.columns)
    txt_imp = imp[[c for c in TEXT_FEATURES if c in imp.index]].sort_values(ascending=False)
    print(f"\n  text-feature importance (share of total: {100*txt_imp.sum():.1f}%):")
    for name, v in txt_imp.head(8).items():
        if v > 0:
            print(f"    {name:<16}{v:.4f}")

    verdict = ("INTEGRATE — meaningful PR-AUC lift" if d_ap >= 0.02 else
               "MARGINAL — consider embeddings (Path B) before integrating" if d_ap > 0.005 else
               "SKIP for the model — keep keywords as a display-only signal")
    print(f"\n  verdict: {verdict}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)