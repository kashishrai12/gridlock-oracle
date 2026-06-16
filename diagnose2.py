import pandas as pd, numpy as np, sys
df = pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "data/flipkart_gridlock.csv")
s = pd.to_datetime(df["start_datetime"], errors="coerce")

for c in ["closed_datetime","modified_datetime","created_date"]:
    if c in df.columns:
        print(f"[{c}] non-null={df[c].notna().sum()}/{len(df)}  sample={df[c].dropna().astype(str).head(2).tolist()}")
    else:
        print(f"[{c}] MISSING")

# Build best end signal: resolved -> closed -> end
def col(name): 
    return pd.to_datetime(df[name], errors="coerce") if name in df.columns else pd.Series(pd.NaT, index=df.index)
best = col("resolved_datetime").fillna(col("closed_datetime")).fillna(col("end_datetime"))
d = (best - s).dt.total_seconds()/60.0
ok = d.notna() & (d>0)
print(f"\nUsable durations (resolved||closed||end): {ok.sum()}/{len(df)}")
print("\nDuration percentiles (minutes):")
for p in [10,25,50,75,90,95,99]:
    print(f"  p{p:<2}: {np.nanpercentile(d[ok], p):,.0f} min  ({np.nanpercentile(d[ok], p)/60:,.1f} h)")
# cross-check against status
if "status" in df.columns:
    tmp = pd.DataFrame({"status":df["status"],"ok":ok})
    print("\nusable durations by status:\n", tmp[tmp.ok].status.value_counts())