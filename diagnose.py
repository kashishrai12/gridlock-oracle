import pandas as pd, numpy as np, sys
df = pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "data/flipkart_gridlock.csv")
print(f"rows: {len(df)}   cols: {len(df.columns)}")
print("\n--- columns ---"); print(list(df.columns))

for c in ["start_datetime","end_datetime","resolved_datetime","status","priority","event_type"]:
    if c in df.columns:
        nonnull = df[c].notna().sum()
        blank = (df[c].astype(str).str.strip()=="").sum()
        print(f"\n[{c}] non-null={nonnull}/{len(df)}  empty-string={blank}")
        print("  sample:", df[c].dropna().astype(str).head(3).tolist())
    else:
        print(f"\n[{c}] MISSING")

# how many usable durations from each end signal?
def usable(endcol):
    if endcol not in df.columns: return "col missing"
    s = pd.to_datetime(df["start_datetime"], errors="coerce")
    e = pd.to_datetime(df[endcol], errors="coerce")
    d = (e-s).dt.total_seconds()/60
    return int(((d.notna()) & (d>0)).sum())
print("\n--- usable (>0 min) durations ---")
print("  from resolved_datetime:", usable("resolved_datetime"))
print("  from end_datetime     :", usable("end_datetime"))
# combined: prefer resolved, fallback end
s = pd.to_datetime(df["start_datetime"], errors="coerce")
r = pd.to_datetime(df.get("resolved_datetime"), errors="coerce") if "resolved_datetime" in df else pd.Series(pd.NaT,index=df.index)
e = pd.to_datetime(df.get("end_datetime"), errors="coerce") if "end_datetime" in df else pd.Series(pd.NaT,index=df.index)
best = r.fillna(e)
d = (best-s).dt.total_seconds()/60
print("  combined (resolved||end):", int(((d.notna())&(d>0)).sum()))
if "status" in df.columns: print("\nstatus counts:\n", df["status"].value_counts())