"""
Decision tool: where is the real signal in this data?
Tests, on a TEMPORAL split (honest), three candidate targets and prints whether
each beats its naive baseline. Also runs a random-split regression to tell
no-signal apart from temporal-drift.
Run: python diagnose3.py --data data/flipkart_gridlock.csv
"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import r2_score, roc_auc_score, f1_score, accuracy_score
import xgboost as xgb
from utils import preprocess as pp

def fit_eval_reg(Xtr,ytr,Xte,yte):
    m=xgb.XGBRegressor(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.9,
        colsample_bytree=0.9,reg_lambda=1.0,tree_method="hist",enable_categorical=True,
        random_state=42,n_jobs=-1)
    m.fit(Xtr,ytr,verbose=False); return r2_score(yte,m.predict(Xte))

def main(path):
    df=pp.load_and_prepare(path)
    tr,te=pp.temporal_split(df,0.8)
    stats=pp.fit_location_stats(tr)
    Xtr,ytr,trd=pp.build_xy(tr,stats); Xte,yte,ted=pp.build_xy(te,stats)

    print("\n================ SIGNAL LOCATOR ================")

    # --- 1) clearance regression: temporal vs random (drift vs no-signal) ---
    r2_temporal=fit_eval_reg(Xtr,ytr,Xte,yte)
    dfx=df.sample(frac=1.0,random_state=0).reset_index(drop=True)
    cut=int(len(dfx)*0.8); rtr,rte=dfx.iloc[:cut],dfx.iloc[cut:]
    rstats=pp.fit_location_stats(rtr)
    RXtr,Rytr,_=pp.build_xy(rtr,rstats); RXte,Ryte,_=pp.build_xy(rte,rstats)
    r2_random=fit_eval_reg(RXtr,Rytr,RXte,Ryte)
    print(f"\n[1] CLEARANCE REGRESSION  R2(log)")
    print(f"    temporal split : {r2_temporal:+.3f}   (honest / production-like)")
    print(f"    random   split : {r2_random:+.3f}   (in-distribution ceiling)")
    verdict = ("no signal" if r2_random<0.05 else
               "TEMPORAL DRIFT (model learns, but patterns shift over time)" if r2_temporal<0.05
               else "usable")
    print(f"    -> {verdict}")

    # --- 2) closure-need classification (drives barricading) ---
    # exclude closure-derived features to avoid leakage
    drop=[c for c in ["closure_int","junction_closure_rate","corridor_closure_rate"] if c in Xtr]
    Xtr2,Xte2=Xtr.drop(columns=drop),Xte.drop(columns=drop)
    ytr2,yte2=trd["closure_int"].astype(int),ted["closure_int"].astype(int)
    if yte2.nunique()>1 and ytr2.nunique()>1:
        c=xgb.XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.9,
            colsample_bytree=0.9,tree_method="hist",enable_categorical=True,
            eval_metric="logloss",random_state=42,n_jobs=-1)
        c.fit(Xtr2,ytr2,verbose=False)
        auc=roc_auc_score(yte2,c.predict_proba(Xte2)[:,1])
        base=max(yte2.mean(),1-yte2.mean())
        print(f"\n[2] CLOSURE-NEED CLASSIFICATION (barricading)")
        print(f"    positive rate={yte2.mean():.1%}  AUC={auc:.3f}  (0.5=useless, >0.7=useful)")
    else:
        print("\n[2] CLOSURE-NEED: only one class present, skipped")

    # --- 3) clearance BUCKET classification (fast/med/slow) ---
    q1,q2=trd[pp.TARGET_RAW].quantile([0.4,0.75]).values
    def bucket(x): return 0 if x<=q1 else (1 if x<=q2 else 2)
    ybt=trd[pp.TARGET_RAW].map(bucket); ybe=ted[pp.TARGET_RAW].map(bucket)
    cb=xgb.XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.9,
        colsample_bytree=0.9,tree_method="hist",enable_categorical=True,
        eval_metric="mlogloss",random_state=42,n_jobs=-1)
    cb.fit(Xtr,ybt,verbose=False); pb=cb.predict(Xte)
    acc=accuracy_score(ybe,pb); f1=f1_score(ybe,pb,average="macro")
    base_acc=ybe.value_counts(normalize=True).max()
    print(f"\n[3] CLEARANCE BUCKET (fast<{q1:.0f}m / med / slow>{q2:.0f}m)")
    print(f"    accuracy={acc:.3f} (baseline {base_acc:.3f})  macro-F1={f1:.3f}")
    print("\n===============================================")
    print("READ: if [2] AUC>0.7 -> barricading classifier is your headline.")
    print("      if [3] beats baseline by >8pts -> bucket model is viable.")
    print("      if [1] random>>temporal -> it's drift; report bucket/closure instead.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True)
    main(ap.parse_args().data)