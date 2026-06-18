"""
ab_test_embeddings.py — Path B: do MULTILINGUAL EMBEDDINGS of `description` improve the
closure model, where keyword features did not?

Keywords only capture coarse categories you already have structured. Embeddings capture
nuance ("minor breakdown cleared quickly" vs "lorry overturned blocking both lanes") and
handle English + Kannada automatically. We:
  1. encode descriptions with a pretrained multilingual sentence model (no fitting on our
     data -> no leakage), or fall back to char-ngram TF-IDF + SVD (script-agnostic, no torch),
  2. reduce dimensionality with PCA/SVD FIT ON TRAIN ONLY,
  3. run the same honest A/B as the keyword test and report ROC-AUC / PR-AUC + lift.

Primary path needs: pip install sentence-transformers
Fallback path needs nothing new (uses scikit-learn you already have).

Run: python ab_test_embeddings.py --data data/flipkart_gridlock.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import preprocess as pp

ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # supports Kannada + English
N_COMPONENTS = 24


def get_desc(df):
    s = df["description"].astype(str).fillna("") if "description" in df.columns \
        else pd.Series([""] * len(df), index=df.index)
    return s.replace({"nan": "", "None": "", "none": ""}).tolist()


def embed(tr_text, te_text):
    """Return (emb_tr, emb_te, method_name). Train-only fitting where applicable."""
    # ---- primary: pretrained multilingual sentence-transformer ----
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(ST_MODEL)
        allv = m.encode(tr_text + te_text, batch_size=64, convert_to_numpy=True,
                        show_progress_bar=True, normalize_embeddings=True)
        n = len(tr_text)
        e_tr, e_te = allv[:n], allv[n:]
        # PCA reduce (fit on train only)
        k = min(N_COMPONENTS, e_tr.shape[1], max(2, e_tr.shape[0] - 1))
        pca = PCA(n_components=k, random_state=0).fit(e_tr)
        return pca.transform(e_tr), pca.transform(e_te), f"{ST_MODEL} + PCA{k}"
    except Exception as ex:
        import traceback
        traceback.print_exc()

        print(
            f"[info] sentence-transformers failed ({type(ex).__name__}: {ex}); "
            "using char-ngram fallback."
        )

    # ---- fallback: char-ngram TF-IDF + SVD (script-agnostic, fit on TRAIN only) ----
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=4000)
    Xtr = vec.fit_transform(tr_text)
    Xte = vec.transform(te_text)
    k = min(32, Xtr.shape[1] - 1) if Xtr.shape[1] > 2 else 2
    svd = TruncatedSVD(n_components=k, random_state=0).fit(Xtr)
    return svd.transform(Xtr), svd.transform(Xte), f"tfidf-char(2-4)+svd{k} (fallback)"


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

    # baseline
    auc0, ap0, _ = fit_eval(Xtr, ytr, Xte, yte)

    # embeddings
    e_tr, e_te, method = embed(get_desc(train_df), get_desc(test_df))
    emb_cols = [f"emb_{i}" for i in range(e_tr.shape[1])]
    Etr = pd.DataFrame(e_tr, columns=emb_cols, index=Xtr.index)
    Ete = pd.DataFrame(e_te, columns=emb_cols, index=Xte.index)
    Xtr2 = pd.concat([Xtr, Etr], axis=1)
    Xte2 = pd.concat([Xte, Ete], axis=1)
    auc1, ap1, clf1 = fit_eval(Xtr2, ytr, Xte2, yte)

    base = yte.mean()
    print(f"\n===== A/B: do multilingual EMBEDDINGS help? (test {len(yte)} events, "
          f"{int(yte.sum())} closures, base {base:.1%}) =====")
    print(f"  embedding method: {method}\n")
    print(f"{'variant':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'PR lift':>10}")
    print(f"{'structured only':<26}{auc0:>9.3f}{ap0:>9.3f}{ap0/base:>8.2f}x")
    print(f"{'structured + embeddings':<26}{auc1:>9.3f}{ap1:>9.3f}{ap1/base:>8.2f}x")
    d_auc, d_ap = auc1 - auc0, ap1 - ap0
    print(f"\n  delta ROC-AUC: {d_auc:+.3f}   delta PR-AUC: {d_ap:+.3f}")

    imp = pd.Series(clf1.feature_importances_, index=Xtr2.columns)
    emb_share = imp[emb_cols].sum()
    print(f"  embedding-feature importance share: {100*emb_share:.1f}%")

    verdict = ("INTEGRATE — embeddings add real lift" if d_ap >= 0.02 else
               "MARGINAL — small lift; judge effort vs deadline" if d_ap > 0.005 else
               "SKIP — text doesn't help the model; NLP question is closed with evidence")
    print(f"\n  verdict: {verdict}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)