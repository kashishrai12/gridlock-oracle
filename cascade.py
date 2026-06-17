"""
cascade.py — Compounding-event (cascade) detection.

Insight: when two incidents are active AT THE SAME TIME at the SAME place, their traffic
impact compounds. This finds those overlapping pairs, scores how badly they compound,
and ranks the most cascade-prone locations.

Location key:
  • events with a real corridor          -> grouped by corridor
  • events tagged "Non-corridor"/blank   -> grouped by a ~330 m spatial cell (lat/lon),
    so genuine same-spot overlaps are caught but far-apart events are never lumped.

Standalone build:  python cascade.py --data data/flipkart_gridlock.csv
Dashboard use:     import cascade; cascade.render_cascade_page()
"""

import argparse, os
import numpy as np
import pandas as pd

MODELS_DIR = "models"
CASCADE_PATH = f"{MODELS_DIR}/cascades.csv"
DEFAULT_DUR_MIN = 60
CAP_DUR_MIN = 1440
GEO_CELL = 0.003
INVALID_CORRIDORS = {"non-corridor", "noncorridor", "unknown", "none", "na",
                     "nan", "", "-", "null"}


def _load_events(path):
    df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df = df[df["start"].notna()].copy()

    def col(c):
        return pd.to_datetime(df[c], errors="coerce", utc=True) if c in df.columns \
            else pd.Series(pd.NaT, index=df.index)
    end = col("resolved_datetime").fillna(col("closed_datetime")).fillna(col("end_datetime"))
    dur = (end - df["start"]).dt.total_seconds() / 60
    end = end.where((dur > 0) & (dur <= CAP_DUR_MIN))
    df["end"] = end.fillna(df["start"] + pd.Timedelta(minutes=DEFAULT_DUR_MIN))

    if "requires_road_closure" in df.columns:
        df["closure"] = (df["requires_road_closure"].astype(str).str.lower()
                         .isin(["1", "true", "yes", "y", "t"]).astype(int))
    else:
        df["closure"] = 0
    if "id" not in df.columns:
        df["id"] = range(len(df))

    corr = (df["corridor"].astype(str).str.strip() if "corridor" in df.columns
            else pd.Series("", index=df.index))
    valid = ~corr.str.lower().isin(INVALID_CORRIDORS)
    df["group_type"] = np.where(valid, "corridor", "geo")
    df["location"] = np.where(valid, corr, np.nan)

    if {"latitude", "longitude"}.issubset(df.columns):
        la = (df["latitude"] / GEO_CELL).round() * GEO_CELL
        lo = (df["longitude"] / GEO_CELL).round() * GEO_CELL
        geo_label = "geo " + la.round(3).astype(str) + "," + lo.round(3).astype(str)
        fill = (~valid) & df["latitude"].notna() & df["longitude"].notna()
        df.loc[fill, "location"] = geo_label[fill]

    df = df[df["location"].notna() & (df["location"].astype(str) != "nan")].copy()
    return df


def _cascade_risk(overlap_min, both_closure):
    s = 2.0 + 3.0 * min(overlap_min / 120.0, 1.0) + 2.5 * both_closure
    return round(min(s, 10.0), 1)


def detect_cascades(path):
    df = _load_events(path)
    rows = []
    for loc, grp in df.groupby("location"):
        g = grp.sort_values("start")
        starts = g["start"].values; ends = g["end"].values
        ids = g["id"].values; clo = g["closure"].values; gtype = g["group_type"].values
        n = len(g)
        for a in range(n):
            for b in range(a + 1, n):
                if starts[b] >= ends[a]:
                    break
                ov_start = max(starts[a], starts[b]); ov_end = min(ends[a], ends[b])
                ov_min = (ov_end - ov_start) / np.timedelta64(1, "m")
                if ov_min <= 0:
                    continue
                both_clo = int(clo[a]) + int(clo[b])
                rows.append({"location": loc, "group_type": gtype[a],
                             "event_a": ids[a], "event_b": ids[b],
                             "overlap_start": pd.Timestamp(ov_start),
                             "overlap_end": pd.Timestamp(ov_end),
                             "overlap_min": round(float(ov_min), 1),
                             "closures_in_pair": both_clo,
                             "cascade_risk": _cascade_risk(ov_min, both_clo)})
    return pd.DataFrame(rows), df


def build_cascades(path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    cas, df = detect_cascades(path)
    cas.to_csv(CASCADE_PATH, index=False)
    involved = len(set(cas["event_a"]).union(set(cas["event_b"]))) if len(cas) else 0
    print(f"[cascade] {len(df)} events grouped into {df['location'].nunique()} locations")
    print(f"[cascade] {len(cas)} overlapping pairs; "
          f"{involved} events ({involved/max(len(df),1):.0%}) involved")
    if len(cas):
        print(f"[cascade] by location type: {cas['group_type'].value_counts().to_dict()}")
        top = cas.groupby("location").size().sort_values(ascending=False).head(1)
        print(f"[cascade] most cascade-prone: {top.index[0]} ({int(top.iloc[0])} pairs)")
        print(f"[cascade] saved {CASCADE_PATH}")
    return cas


def render_cascade_page(data_path="data/flipkart_gridlock.csv"):
    import streamlit as st
    import plotly.express as px

    st.title("⚠️ Event Cascade Detection")
    st.caption("Concurrent incidents at the same location compound — combined impact is "
               "worse than the sum of parts. Real corridors grouped by corridor; untagged "
               "events by ~330 m spatial proximity.")

    if os.path.exists(CASCADE_PATH):
        cas = pd.read_csv(CASCADE_PATH, parse_dates=["overlap_start", "overlap_end"])
    elif os.path.exists(data_path):
        with st.spinner("Detecting cascades..."):
            cas = build_cascades(data_path)
    else:
        st.error("Run `python cascade.py --data data/flipkart_gridlock.csv` first.")
        return

    if cas is None or len(cas) == 0:
        st.info("No overlapping events detected."); return

    c1, c2, c3 = st.columns(3)
    c1.metric("Cascade pairs", len(cas))
    c2.metric("High-risk (≥7)", int((cas["cascade_risk"] >= 7).sum()))
    c3.metric("Locations affected", cas["location"].nunique())

    view = st.radio("Show", ["Real corridors", "Spatial clusters", "All"],
                    horizontal=True, index=0)
    show = (cas[cas["group_type"] == "corridor"] if view == "Real corridors"
            else cas[cas["group_type"] == "geo"] if view == "Spatial clusters" else cas)
    if len(show) == 0:
        st.info("No cascades of this type."); return

    st.subheader("Most cascade-prone locations")
    top = (show.groupby("location")
             .agg(pairs=("cascade_risk", "size"), avg_risk=("cascade_risk", "mean"))
             .sort_values("pairs", ascending=False).head(12).reset_index())
    fig = px.bar(top.iloc[::-1], x="pairs", y="location", orientation="h",
                 color="avg_risk", color_continuous_scale="OrRd",
                 labels={"pairs": "overlapping pairs", "avg_risk": "avg risk"})
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Worst cascades")
    worst = show.sort_values("cascade_risk", ascending=False).head(15).copy()
    worst["window"] = (worst["overlap_start"].dt.strftime("%Y-%m-%d %H:%M")
                       + " → " + worst["overlap_end"].dt.strftime("%H:%M"))
    st.dataframe(worst[["location", "window", "overlap_min", "closures_in_pair", "cascade_risk"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Cascade timeline — most-affected location")
    focus = top.iloc[0]["location"]
    sub = show[show["location"] == focus].sort_values("cascade_risk", ascending=False).head(30)
    if len(sub):
        tl = pd.DataFrame([{"Pair": f"{r.event_a}+{r.event_b}", "Start": r.overlap_start,
                            "End": r.overlap_end, "Risk": r.cascade_risk} for r in sub.itertuples()])
        fig = px.timeline(tl, x_start="Start", x_end="End", y="Pair", color="Risk",
                          color_continuous_scale="OrRd", title=f"Location: {focus}")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=420, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    build_cascades(ap.parse_args().data)