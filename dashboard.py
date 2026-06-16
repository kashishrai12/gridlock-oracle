"""
dashboard.py  —  Gridlock Oracle (rewired for the closure-need model)

Run: streamlit run dashboard.py
Prereqs: python train_model.py --data data/flipkart_gridlock.csv
         python hotspots.py   --data data/flipkart_gridlock.csv

Pages:
  1. Predict Event       closure probability + impact + resource plan + explanation
  2. Risk Map            zone x hour hotspot heatmap + junction leaderboard
  3. Analytics           temporal / cause / closure-rate analytics (clearance = descriptive)
  4. Diversion & Barricades   routing_page (unchanged)
  5. Learning Loop       feedback on closure-prediction correctness
"""

import os, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from predictor import GridlockPredictor

MODELS_DIR = "models"
st.set_page_config(page_title="Gridlock Oracle", layout="wide", page_icon="🚦")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_predictor():
    return GridlockPredictor(MODELS_DIR)


@st.cache_data
def load_enriched():
    p = f"{MODELS_DIR}/enriched_dataset.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_csv(name):
    p = f"{MODELS_DIR}/{name}"
    return pd.read_csv(p) if os.path.exists(p) else None


def options(df, col, limit=200):
    if df is None or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str)
    return sorted(vals.value_counts().head(limit).index.tolist())


def artifacts_ready():
    need = ["closure_clf.json", "location_stats.pkl", "enriched_dataset.csv"]
    return all(os.path.exists(f"{MODELS_DIR}/{n}") for n in need)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🚦 Gridlock Oracle")
st.sidebar.caption("Event-driven congestion intelligence")
page = st.sidebar.radio("Navigate", [
    "Predict Event", "Risk Map", "Analytics", "Diversion & Barricades", "Learning Loop"])

if not artifacts_ready():
    st.error("Model artifacts not found. Run `python train_model.py --data data/flipkart_gridlock.csv` "
             "and `python hotspots.py --data data/flipkart_gridlock.csv` first.")
    st.stop()

enriched = load_enriched()


# --------------------------------------------------------------------------- #
# 1) PREDICT EVENT
# --------------------------------------------------------------------------- #
if page == "Predict Event":
    st.header("Predict Event Impact")
    st.caption("Closure-need probability drives the barricading & resource recommendation.")

    c1, c2, c3 = st.columns(3)
    with c1:
        cause = st.selectbox("Event cause", options(enriched, "event_cause") or ["Accident"])
        veh = st.selectbox("Vehicle type", options(enriched, "veh_type") or ["Car"])
    with c2:
        junction = st.selectbox("Junction", options(enriched, "junction") or ["Unknown"])
        zone = st.selectbox("Zone", options(enriched, "zone") or ["Unknown"])
    with c3:
        corridor = st.selectbox("Corridor", options(enriched, "corridor") or ["Unknown"])
        ps = st.selectbox("Police station", options(enriched, "police_station") or ["Unknown"])

    c4, c5, c6 = st.columns(3)
    with c4:
        priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], index=2)
    with c5:
        etype = st.selectbox("Type", ["unplanned", "planned"])
    with c6:
        when = st.time_input("Time of day", dt.time(18, 0))

    if st.button("Predict", type="primary"):
        predictor = load_predictor()
        event = {
            "start_datetime": f"2024-06-01 {when.strftime('%H:%M')}",
            "priority": priority, "event_type": etype, "event_cause": cause,
            "veh_type": veh, "junction": junction, "corridor": corridor,
            "zone": zone, "police_station": ps,
        }
        r = predictor.predict(event)

        g1, g2 = st.columns([1, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=r["closure_prob"] * 100,
                number={"suffix": "%"},
                title={"text": "Road-closure probability"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#d62728" if r["closure_prob"] >= 0.5 else "#1f77b4"},
                       "steps": [{"range": [0, 30], "color": "#eaf3ea"},
                                 {"range": [30, 60], "color": "#fff3cd"},
                                 {"range": [60, 100], "color": "#f8d7da"}]}))
            gauge.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        with g2:
            st.metric("Impact score", f"{r['impact_score']} / 10", r["impact_tier"])
            st.metric("Expected clearance (historical)", f"{r['expected_clearance_mins']} min")
            if not r["is_known_location"]:
                st.warning("New location — estimate uses zone/global fallback.")

        st.subheader("Recommended response")
        res = r["resources"]
        m = st.columns(5)
        m[0].metric("Barricade?", "YES" if res["barricading_recommended"] else "no")
        m[1].metric("Barricades", res["barricades"])
        m[2].metric("Personnel", res["personnel"])
        m[3].metric("Supervisors", res["supervisors"])
        m[4].metric("Rapid response", "YES" if res["rapid_response_required"] else "no")

        st.subheader("Why this prediction")
        ex = pd.DataFrame(r["explanations"])
        if not ex.empty:
            ex["signed"] = ex["contribution"]
            fig = px.bar(ex.iloc[::-1], x="signed", y="feature", orientation="h",
                         color="signed", color_continuous_scale=["#1f77b4", "#d62728"],
                         labels={"signed": "contribution to closure risk"})
            fig.update_layout(height=300, showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # map pin from junction's mean coords
        if enriched is not None and {"latitude", "longitude"}.issubset(enriched.columns):
            jrows = enriched[enriched["junction"].astype(str) == str(junction)]
            if not jrows.empty:
                st.map(pd.DataFrame({"lat": [jrows["latitude"].mean()],
                                     "lon": [jrows["longitude"].mean()]}), zoom=12)


# --------------------------------------------------------------------------- #
# 2) RISK MAP  (hotspots)
# --------------------------------------------------------------------------- #
elif page == "Risk Map":
    st.header("Hotspot Risk Map")
    zh = load_csv("hotspot_zone_hour.csv")
    jl = load_csv("hotspot_junctions.csv")
    if zh is None or jl is None:
        st.warning("Run `python hotspots.py --data data/flipkart_gridlock.csv` to generate hotspots.")
        st.stop()

    st.subheader("Event load by zone × hour of day")
    zcol = zh.columns[0]
    mat = zh.set_index(zcol)
    fig = px.imshow(mat, aspect="auto", color_continuous_scale="OrRd",
                    labels=dict(x="Hour of day", y="Zone", color="Events"))
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top incident junctions")
    show = jl.head(15).copy()
    if "closure_rate" in show.columns:
        show["closure_rate"] = (show["closure_rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(show, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 3) ANALYTICS
# --------------------------------------------------------------------------- #
elif page == "Analytics":
    st.header("Analytics")
    hourly = load_csv("hotspot_hourly.csv")
    dow = load_csv("hotspot_dow.csv")

    a, b = st.columns(2)
    with a:
        if hourly is not None:
            fig = px.bar(hourly, x="hour", y="events", title="Events by hour of day")
            fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)
    with b:
        if dow is not None:
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dd = dow.copy(); dd["day"] = dd["day_of_week"].map(lambda i: names[int(i)])
            fig = px.bar(dd, x="day", y="events", title="Events by day of week")
            fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)

    if enriched is not None:
        c, d = st.columns(2)
        with c:
            if "event_cause" in enriched.columns:
                vc = enriched["event_cause"].value_counts().head(12).reset_index()
                vc.columns = ["event_cause", "count"]
                fig = px.bar(vc, x="count", y="event_cause", orientation="h",
                             title="Events by cause")
                fig.update_layout(height=360, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
        with d:
            if {"event_cause", "closure_int"}.issubset(enriched.columns):
                cr = (enriched.groupby("event_cause")["closure_int"].mean()
                      .sort_values(ascending=False).head(12).reset_index())
                cr["closure_int"] *= 100
                fig = px.bar(cr, x="closure_int", y="event_cause", orientation="h",
                             title="Closure rate by cause (%)")
                fig.update_layout(height=360, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        if "clearance_mins" in enriched.columns:
            st.subheader("Clearance time (descriptive — not predicted)")
            cl = enriched[enriched["clearance_mins"].notna()]
            fig = px.histogram(cl, x="clearance_mins", nbins=40,
                               title="Historical clearance-time distribution")
            fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 4) DIVERSION & BARRICADES  (unchanged routing page)
# --------------------------------------------------------------------------- #
elif page == "Diversion & Barricades":
    st.header("Diversion & Barricades")
    try:
        import routing_page
        # the entry function may be named anything — find it
        candidates = ["render_routing_page", "render", "main", "show", "app",
                      "render_page", "display", "diversion_barricades",
                      "routing_page", "page", "run", "show_page"]
        entry = next((getattr(routing_page, n) for n in candidates
                      if callable(getattr(routing_page, n, None))), None)
        if entry is not None:
            entry()
        else:
            funcs = [n for n in dir(routing_page)
                     if callable(getattr(routing_page, n)) and not n.startswith("_")]
            st.warning("Couldn't auto-detect the routing page's entry function.")
            st.write("Functions available in routing_page.py:", funcs)
            st.caption("Replace `entry()` with the correct one from this list.")
    except Exception as e:
        st.error(f"Routing page failed to load: {e}")


# --------------------------------------------------------------------------- #
# 5) LEARNING LOOP
# --------------------------------------------------------------------------- #
elif page == "Learning Loop":
    st.header("Learning Loop")
    st.caption("Log whether a closure was actually needed, to refine the model over time.")
    log_path = f"{MODELS_DIR}/feedback_log.csv"

    with st.form("fb"):
        junction = st.selectbox("Junction", options(enriched, "junction") or ["Unknown"])
        prob = st.slider("Predicted closure probability", 0.0, 1.0, 0.5, 0.01)
        actual = st.radio("Was a road closure actually required?", ["Yes", "No"], horizontal=True)
        submitted = st.form_submit_button("Submit feedback")
    if submitted:
        row = {"timestamp": dt.datetime.now().isoformat(), "junction": junction,
               "predicted_closure_prob": prob, "actual_closure": int(actual == "Yes")}
        df = pd.DataFrame([row])
        df.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
        st.success("Logged. Retraining triggers automatically once ≥50 new records accumulate.")

    if os.path.exists(log_path):
        fb = pd.read_csv(log_path)
        st.metric("Feedback records", len(fb))
        if len(fb) >= 2:
            fig = px.scatter(fb, x="predicted_closure_prob", y="actual_closure",
                             title="Predicted probability vs actual outcome")
            fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)