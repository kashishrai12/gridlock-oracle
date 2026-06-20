
import os, datetime as dt
import pandas as pd
import streamlit as st
import plotly.express as px

from predictor import GridlockPredictor

MODELS_DIR = "models"
st.set_page_config(page_title="Gridlock Oracle", layout="wide", page_icon="🚦")

import os
try:
    import streamlit as st
    for _k in ["TOMTOM_API_KEY", "MAPPLS_CLIENT_ID", "MAPPLS_CLIENT_SECRET", "MAPPLS_REST_KEY"]:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Theme — every color the app uses, light and dark, in one place
# --------------------------------------------------------------------------- #
THEME_LIGHT = {
    "bg": "#F3F4F8", "surface": "#FFFFFF", "surface-alt": "#FAFAFD",
    "border": "#E7E8F0", "ink": "#15171F", "muted": "#3B3E46",
    "primary": "#4338CA", "primary-soft": "#3C3B4B", "on-primary": "#1A1515",
    "green": "#16A34A", "green-soft": "#00FF62",
    "amber": "#D97706", "amber-soft": "#F0AD39",
    "red": "#DC2626", "red-soft": "#F86C6C",
    "shadow": "0 1px 2px rgba(16,24,40,.04), 0 10px 28px rgba(16,24,40,.06)",
}
THEME_DARK = {
    "bg": "#0E1015", "surface": "#171A21", "surface-alt": "#12141A",
    "border": "#272B35", "ink": "#EDEEF3", "muted": "#919AAA",
    "primary": "#7C75F0", "primary-soft": "rgba(124,117,240,.16)", "on-primary": "#0E1015",
    "green": "#34D399", "green-soft": "rgba(52,211,153,.14)",
    "amber": "#FBBF24", "amber-soft": "rgba(251,191,36,.14)",
    "red": "#F87171", "red-soft": "rgba(248,113,113,.14)",
    "shadow": "0 1px 2px rgba(0,0,0,.45), 0 10px 28px rgba(0,0,0,.5)",
}

APP_CSS_STATIC = """
*, *::before, *::after { box-sizing: border-box; }
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.4rem; max-width: 1600px; }
#MainMenu, header[data-testid="stHeader"] { background: transparent; }
/* Streamlit labels */
label,
.stCheckbox label,
.stRadio label,
.stSlider label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label {
    color: var(--ink) !important;
    opacity: 1 !important;
}

/* Markdown text */
[data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}

[data-testid="stMarkdownContainer"] * {
    color: inherit !important;
}

/* Headers */
h1,h2,h3,h4,h5,h6,
[data-testid="stHeading"] {
    color: var(--ink) !important;
}
.stButton button {
    color: var(--ink) !important;
}
/* ---------------------------------------------------------------- */
/* Sidebar                                                            */
/* ---------------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }

.brand-row { display:flex; align-items:center; gap:.6rem; padding:.4rem 0 1rem; }
.brand-mark { width:36px; height:36px; border-radius:10px; background:var(--primary);
              display:flex; align-items:center; justify-content:center; color:var(--on-primary); flex:none; }
.brand-name { font-weight:800; font-size:1.05rem; color:var(--ink); line-height:1.1; }
.brand-tag { font-size:.72rem; color:var(--muted); margin-top:.1rem; }

.nav-group-label { font-size:.68rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
                    color:var(--muted); margin:1.1rem 0 .3rem .2rem; }

[data-testid="stSidebar"] .stButton button {
  width:100%; text-align:left; justify-content:flex-start;
  border-radius:10px; padding:.55rem .7rem; font-weight:600; font-size:.92rem;
  border:1px solid transparent !important; box-shadow:none !important;
}
[data-testid="stSidebar"] .stButton button[kind="secondary"] {
  background:transparent !important; color:var(--ink) !important;
}
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
  background:var(--primary-soft) !important; color:var(--primary) !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background:var(--primary) !important; color:var(--on-primary) !important; border-color:var(--primary) !important;
}

/* ---------------------------------------------------------------- */
/* Top bar                                                            */
/* ---------------------------------------------------------------- */
.topbar-eyebrow { font-size:.74rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--primary); }
.topbar-title { font-size:1.9rem; font-weight:800; color:var(--ink); margin:.15rem 0 .3rem; line-height:1.15; }
.topbar-desc { color:var(--muted); font-size:.92rem; margin:0 0 .2rem; max-width:62ch; }
.meta-row { font-size:.78rem; color:var(--muted); margin-bottom:1.1rem; }

div[data-testid="column"]:has(button[key="theme_toggle"]) { display:flex; align-items:flex-start; justify-content:flex-end; }
.stButton button[kind="secondary"]#, button[kind="secondary"] { }
button[data-testid="baseButton-secondary"]:has(div:contains("Dark")) { }
.stButton button { border-radius:999px; }

/* ---------------------------------------------------------------- */
/* Card shell (pure-HTML cards: leaderboard, risk meter, alerts)     */
/* ---------------------------------------------------------------- */
.card { background:var(--surface); border:1px solid var(--border); border-radius:18px;
        padding:1.15rem 1.3rem; box-shadow:var(--shadow); margin-bottom:1.1rem; }
.card-header { display:flex; align-items:flex-start; gap:.7rem; margin-bottom:.85rem; }
.card-icon { width:34px; height:34px; border-radius:9px; background:var(--primary-soft); color:var(--primary);
             display:flex; align-items:center; justify-content:center; flex:none; }
.card-title { font-weight:700; font-size:.98rem; color:var(--ink); }
.card-sub { font-size:.78rem; color:var(--muted); margin-top:.1rem; }
.empty-note { font-size:.86rem; color:var(--muted); padding:.4rem 0; }

/* container(border=True) cards — best-effort selectors, Streamlit version dependent */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stContainer"] {
  background:var(--surface) !important; border:1px solid var(--border) !important;
  border-radius:18px !important; box-shadow:var(--shadow); margin-bottom:1.1rem;
}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stContainer"] {
  padding:1.15rem 1.3rem; box-sizing:border-box;
}
[data-testid="stForm"] { background:var(--surface) !important; border:1px solid var(--border) !important;
  border-radius:18px !important; box-shadow:var(--shadow); padding:1.15rem 1.3rem !important; margin-bottom:1.1rem; }

/* ---------------------------------------------------------------- */
/* Stat cards                                                        */
/* ---------------------------------------------------------------- */
.stat-grid { display:flex; gap:.85rem; flex-wrap:wrap; margin-bottom:1.1rem; }
.stat-card { flex:1 1 150px; background:var(--surface); border:1px solid var(--border); border-radius:16px;
             padding:.95rem 1.05rem; box-shadow:var(--shadow); }
.stat-icon { width:30px; height:30px; border-radius:8px; display:flex; align-items:center; justify-content:center; margin-bottom:.55rem; }
.stat-label { font-size:.74rem; color:var(--muted); font-weight:600; }
.stat-value { font-size:1.55rem; font-weight:800; line-height:1.15; margin-top:.1rem; }
.stat-sub { font-size:.74rem; color:var(--muted); margin-top:.1rem; }

/* ---------------------------------------------------------------- */
/* Risk meter (signature element)                                    */
/* ---------------------------------------------------------------- */
.risk-row { display:flex; align-items:baseline; gap:.65rem; margin-bottom:.7rem; }
.risk-value { font-size:2.5rem; font-weight:800; line-height:1; }
.risk-pill { font-size:.74rem; font-weight:700; padding:.22rem .65rem; border-radius:999px; }
.risk-bar { display:flex; gap:5px; }
.risk-bar .seg { flex:1; height:10px; border-radius:5px; }

/* ---------------------------------------------------------------- */
/* Alerts                                                             */
/* ---------------------------------------------------------------- */
.alert-box { display:flex; gap:.65rem; align-items:flex-start; border:1px solid; border-radius:14px;
             padding:.75rem .95rem; margin:.6rem 0; }
.alert-icon { margin-top:.1rem; flex:none; }
.alert-title { font-weight:700; font-size:.88rem; }
.alert-body { font-size:.84rem; color:var(--muted); margin-top:.15rem; }

/* ---------------------------------------------------------------- */
/* Leaderboard                                                       */
/* ---------------------------------------------------------------- */
.leaderboard { display:flex; flex-direction:column; }
.lb-row { display:grid; grid-template-columns:30px 1fr auto; align-items:center; gap:.8rem;
          padding:.6rem 0; border-bottom:1px solid var(--border); }
.lb-row:last-child { border-bottom:none; }
.lb-rank { font-size:.8rem; font-weight:700; color:var(--muted); }
.lb-name { font-weight:600; color:var(--ink); font-size:.92rem; }
.lb-bar { height:5px; background:var(--bg); border-radius:3px; margin-top:.35rem; overflow:hidden; }
.lb-bar-fill { height:100%; background:var(--primary); }
.lb-meta { display:flex; gap:.5rem; align-items:center; white-space:nowrap; }
.lb-count { font-size:.76rem; color:var(--muted); }
.lb-pill { font-size:.72rem; font-weight:700; padding:.14rem .55rem; border-radius:999px; }

/* ---------------------------------------------------------------- */
/* Native widgets, themed                                            */
/* ---------------------------------------------------------------- */
label, .stSelectbox label, .stSlider label, .stTimeInput label, .stRadio label { color:var(--ink) !important; font-weight:600 !important; font-size:.86rem !important; }
[data-baseweb="select"] > div { background:var(--surface) !important; border-color:var(--border) !important; border-radius:10px !important; color:var(--ink) !important; }
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] { background:var(--surface) !important; color:var(--ink) !important; }
li[role="option"]:hover { background:var(--primary-soft) !important; }
.stTimeInput input, .stNumberInput input, .stTextInput input { background:var(--surface) !important; color:var(--ink) !important; border-color:var(--border) !important; border-radius:10px !important; }
.stSlider [data-baseweb="slider"] { color:var(--primary) !important; }
.stButton button[kind="primary"] { background:var(--primary) !important; border-color:var(--primary) !important; color:var(--on-primary) !important; border-radius:10px !important; }
/**********************/
/* Secondary Buttons  */
/**********************/
.stButton button[kind="secondary"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

.stButton button[kind="secondary"]:hover {
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* force text color */
.stButton button[kind="secondary"] *,
.stButton button[kind="secondary"] span,
.stButton button[kind="secondary"] p,
.stButton button[kind="secondary"] div {
    color: inherit !important;
}
hr { border-color:var(--border) !important; }
"""


def root_vars(theme):
    lines = [f"  --{k}: {v};" for k, v in theme.items()]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


def inject_css(theme):
    st.markdown("<style>" + root_vars(theme) + APP_CSS_STATIC + "</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Tiny hand-drawn icon set (no external icon font / library needed)
# --------------------------------------------------------------------------- #
_ICON_INNER = {
    "octagon": '<polygon points="7,2 17,2 22,7 22,17 17,22 7,22 2,17 2,7"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>'
            '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "bar_chart": '<line x1="5" y1="20" x2="5" y2="11"/><line x1="12" y1="20" x2="12" y2="5"/><line x1="19" y1="20" x2="19" y2="14"/>',
    "gauge": '<path d="M4 16a8 8 0 1 1 16 0"/><line x1="12" y1="16" x2="15" y2="11"/><circle cx="12" cy="16" r="1" fill="currentColor" stroke="none"/>',
    "shield": '<path d="M12 3l7 3v6c0 5-3 7.5-7 9-4-1.5-7-4-7-9V6l7-3z"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><polyline points="12,7 12,12 16,14"/>',
    "users": '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-4 3-6 6-6s6 2 6 6"/><circle cx="17" cy="9" r="2.3"/><path d="M14.2 14.3c1.7.7 2.8 2.6 2.8 5.7"/>',
    "check_circle": '<circle cx="12" cy="12" r="9"/><polyline points="8,12 11,15 16,9"/>',
    "alert_triangle": '<path d="M12 3 2 20h20L12 3z"/><line x1="12" y1="9" x2="12" y2="14"/><circle cx="12" cy="17" r="0.6" fill="currentColor" stroke="none"/>',
    "info": '<circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><circle cx="12" cy="8" r="0.6" fill="currentColor" stroke="none"/>',
    "list": '<line x1="8" y1="6" x2="20" y2="6"/><line x1="8" y1="12" x2="20" y2="12"/><line x1="8" y1="18" x2="20" y2="18"/>'
            '<circle cx="3.5" cy="6" r="1" fill="currentColor" stroke="none"/><circle cx="3.5" cy="12" r="1" fill="currentColor" stroke="none"/>'
            '<circle cx="3.5" cy="18" r="1" fill="currentColor" stroke="none"/>',
    "map_pin": '<path d="M12 21s-7-7.5-7-12a7 7 0 1 1 14 0c0 4.5-7 12-7 12z"/><circle cx="12" cy="9" r="2.4"/>',
    "route": '<path d="M4 4v6a4 4 0 0 0 4 4h8"/><polyline points="12,10 16,14 12,18"/>',
    "loop": '<path d="M3 12a9 9 0 1 1 2.6 6.4"/><polyline points="3,6 3,12 9,12"/>',
}


def icon(name, size=18):
    inner = _ICON_INNER.get(name, _ICON_INNER["info"])
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


# --------------------------------------------------------------------------- #
# Reusable presentation helpers
# --------------------------------------------------------------------------- #
def card_shell(icon_name, title, sub, inner_html):
    return (
        '<div class="card">'
        '<div class="card-header">'
        f'<div class="card-icon">{icon(icon_name, 18)}</div>'
        f'<div><div class="card-title">{title}</div>'
        + (f'<div class="card-sub">{sub}</div>' if sub else "")
        + "</div></div>"
        f'<div>{inner_html}</div></div>'
    )


def card_header(icon_name, title, sub=None):
    """Use at the top of an st.container(border=True) that also holds real widgets/charts."""
    st.markdown(
        '<div class="card-header">'
        f'<div class="card-icon">{icon(icon_name, 18)}</div>'
        f'<div><div class="card-title">{title}</div>'
        + (f'<div class="card-sub">{sub}</div>' if sub else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


TONE = {"neutral": "var(--ink)", "info": "var(--primary)", "good": "var(--green)", "warn": "var(--amber)", "bad": "var(--red)"}
TONE_SOFT = {"neutral": "var(--bg)", "info": "var(--primary-soft)", "good": "var(--green-soft)", "warn": "var(--amber-soft)", "bad": "var(--red-soft)"}
TONE_ICON = {"neutral": "info", "info": "info", "good": "check_circle", "warn": "alert_triangle", "bad": "alert_triangle"}


def stat_card(icon_name, label, value, sub="", tone="neutral"):
    color, soft = TONE[tone], TONE_SOFT[tone]
    return (
        '<div class="stat-card">'
        f'<div class="stat-icon" style="background:{soft};color:{color}">{icon(icon_name, 15)}</div>'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value" style="color:{color}">{value}</div>'
        + (f'<div class="stat-sub">{sub}</div>' if sub else "")
        + "</div>"
    )


def stat_grid(cards):
    st.markdown('<div class="stat-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def alert(tone, title, body=""):
    color, soft = TONE[tone], TONE_SOFT[tone]
    st.markdown(
        f'<div class="alert-box" style="border-color:{color};background:{soft}">'
        f'<div class="alert-icon" style="color:{color}">{icon(TONE_ICON[tone], 16)}</div>'
        f'<div><div class="alert-title" style="color:{color}">{title}</div>'
        + (f'<div class="alert-body">{body}</div>' if body else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def risk_meter(prob):
    if prob >= 0.6:
        tier, label, color, soft = "bad", "High risk", "var(--red)", "var(--red-soft)"
    elif prob >= 0.3:
        tier, label, color, soft = "warn", "Moderate risk", "var(--amber)", "var(--amber-soft)"
    else:
        tier, label, color, soft = "good", "Low risk", "var(--green)", "var(--green-soft)"
    segments, filled = 6, max(1, round(prob * 6))
    segs = "".join(
        f'<span class="seg" style="background:{color if i < filled else "var(--border)"}"></span>'
        for i in range(segments)
    )
    inner = (
        '<div class="risk-row">'
        f'<div class="risk-value" style="color:{color}">{prob * 100:.0f}%</div>'
        f'<div class="risk-pill" style="background:{soft};color:{color}">{label}</div>'
        "</div>"
        f'<div class="risk-bar">{segs}</div>'
    )
    st.markdown(card_shell("gauge", "Closure-need signal", "Probability this event requires a road closure", inner),
                unsafe_allow_html=True)
    return tier


def render_leaderboard(df, rate_col="closure_rate", limit=15):
    if df is None or df.empty:
        st.markdown(card_shell("list", "Top incident junctions", None,
                                '<div class="empty-note">No junction history is available yet.</div>'),
                    unsafe_allow_html=True)
        return
    d = df.head(limit).copy()
    name_col = d.columns[0]
    numeric_cols = [c for c in d.columns if c != rate_col and pd.api.types.is_numeric_dtype(d[c])]
    count_col = numeric_cols[0] if numeric_cols else None
    max_count = d[count_col].max() if count_col else None

    rows = []
    for i, (_, row) in enumerate(d.iterrows(), start=1):
        name = row[name_col]
        count = row[count_col] if count_col else None
        pct = (count / max_count * 100) if (count is not None and max_count) else 0
        meta_bits = []
        if count is not None and pd.notna(count):
            meta_bits.append(f'<span class="lb-count">{count:,.0f} events</span>')
        if rate_col in d.columns and pd.notna(row[rate_col]):
            rv = row[rate_col] * 100 if row[rate_col] <= 1 else row[rate_col]
            tone = "bad" if rv >= 50 else "warn" if rv >= 25 else "good"
            meta_bits.append(
                f'<span class="lb-pill" style="background:{TONE_SOFT[tone]};color:{TONE[tone]}">{rv:.0f}% closure</span>'
            )
        rows.append(
            '<div class="lb-row">'
            f'<div class="lb-rank">{i:02d}</div>'
            '<div>'
            f'<div class="lb-name">{name}</div>'
            f'<div class="lb-bar"><div class="lb-bar-fill" style="width:{pct:.0f}%"></div></div>'
            "</div>"
            f'<div class="lb-meta">{"".join(meta_bits)}</div>'
            "</div>"
        )
    st.markdown(
        card_shell("list", "Top incident junctions", "Ranked by historical event volume",
                   '<div class="leaderboard">' + "".join(rows) + "</div>"),
        unsafe_allow_html=True,
    )


def style_fig(fig, height=320):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color=theme["ink"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=height,
        title=None,
        showlegend=False,
        colorway=[theme["primary"], theme["amber"], theme["red"], theme["muted"]],
    )
    fig.update_xaxes(gridcolor=theme["border"], zerolinecolor=theme["border"], linecolor=theme["border"])
    fig.update_yaxes(gridcolor=theme["border"], zerolinecolor=theme["border"], linecolor=theme["border"])
    return fig


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data
def load_replay():
    import live_feed
    return live_feed.prepare_replay("data/flipkart_gridlock.csv", load_predictor(), max_events=60)

@st.cache_data
def load_conformal():
    import os, pickle
    f = f"{MODELS_DIR}/conformal.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None

@st.cache_data
def load_survival_curves():
    import os
    f = f"{MODELS_DIR}/survival_curves.csv"
    return pd.read_csv(f) if os.path.exists(f) else None

@st.cache_data
def load_survival_params():
    import os, pickle
    f = f"{MODELS_DIR}/survival_params.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None

@st.cache_resource
def load_hawkes():
    import hawkes
    return hawkes.HawkesModel()

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

@st.cache_resource
def load_hawkes():
    import hawkes
    return hawkes.HawkesModel()

@st.cache_data
def load_cascades():
    import cascade
    if os.path.exists(cascade.CASCADE_PATH):
        return pd.read_csv(cascade.CASCADE_PATH, parse_dates=["overlap_start", "overlap_end"])
    if os.path.exists("data/flipkart_gridlock.csv"):
        return cascade.build_cascades("data/flipkart_gridlock.csv")
    return None

@st.cache_data
def load_event_pool(n, seed):
    import optimizer
    return optimizer.build_event_pool(n=n, seed=seed)



@st.cache_data
def load_learning_curve():
    import learning_loop
    return learning_loop.simulate("data/flipkart_gridlock.csv", batches=10)

def options(df, col, limit=200):
    if df is None or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str)
    return sorted(vals.value_counts().head(limit).index.tolist())

NOT_SET = "— not specified —"

def val(selection):
    return "none" if selection == NOT_SET else selection


def artifacts_ready():
    need = ["closure_clf.json", "location_stats.pkl", "enriched_dataset.csv"]
    return all(os.path.exists(f"{MODELS_DIR}/{n}") for n in need)


# --------------------------------------------------------------------------- #
# Theme state + chrome
# --------------------------------------------------------------------------- #
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "page" not in st.session_state:
    st.session_state.page = "Predict Event"

theme = THEME_DARK if st.session_state.dark_mode else THEME_LIGHT
inject_css(theme)

NAV_GROUPS = [
    ("Forecast", [("grid", "Predict Event"), ("bar_chart", "Analytics"), ("alert_triangle", "Event Cascades")]),
    ("Operations", [("route", "Diversion & Barricades"), ("users", "Deployment Optimizer"), ("loop", "Learning Loop"),  ("activity", "Live Feed")]),
]

with st.sidebar:
    st.markdown(
        '<div class="brand-row">'
        f'<div class="brand-mark">{icon("octagon", 18)}</div>'
        '<div><div class="brand-name">Gridlock Oracle</div>'
        '<div class="brand-tag">Congestion intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    for group_label, items in NAV_GROUPS:
        st.markdown(f'<div class="nav-group-label">{group_label}</div>', unsafe_allow_html=True)
        for _icon, name in items:
            active = st.session_state.page == name
            if st.button(name, key=f"nav_{name}", type="primary" if active else "secondary",
                         use_container_width=True):
                st.session_state.page = name

page = st.session_state.page

PAGE_META = {
    "Predict Event": ("Forecast", "Predict event impact",
                       "Closure-need probability drives the barricading & resource recommendation."),
    "Risk Map": ("Surveillance", "Hotspot risk map",
                 "Where and when event load concentrates across the network."),
    "Analytics": ("Patterns", "Analytics",
                  "Temporal, causal, and closure-rate patterns across logged events."),
    "Event Cascades": ("Risk", "Event cascade detection",
                       "Concurrent incidents at the same location compound — combined impact exceeds the sum of parts."),
    "Diversion & Barricades": ("Response", "Diversion & barricades",
                               "Routing console for active closures."),
    "Learning Loop": ("Feedback", "Learning loop",
                       "Log whether a closure was actually needed, to refine the model over time."),
    "Deployment Optimizer": ("Operations", "Deployment optimizer",
                             "Allocate a limited pool of officers and barricades across the day's predicted incidents to mitigate the most disruption."),
    "Live Feed": ("LIVE OPS", "Live incident feed", "Real-time incidents scored as they arrive"),  
    
}

tl, tr = st.columns([6, 1])
with tl:
    eyebrow, title, desc = PAGE_META[page]
    st.markdown(
        f'<div class="topbar-eyebrow">{eyebrow}</div>'
        f'<div class="topbar-title">{title}</div>'
        f'<p class="topbar-desc">{desc}</p>',
        unsafe_allow_html=True,
    )
with tr:
    toggle_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"

    if st.button(
        toggle_label,
        key="theme_toggle",
        type="secondary",
        use_container_width=True
    ):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

if not artifacts_ready():
    alert("bad", "Model artifacts missing",
          "Run <code>python train_model.py --data data/flipkart_gridlock.csv</code> and "
          "<code>python hotspots.py --data data/flipkart_gridlock.csv</code> before loading this dashboard.")
    st.stop()

enriched = load_enriched()
if enriched is not None:
    bits = [dt.date.today().strftime("%d %b %Y")]
    if "junction" in enriched.columns:
        bits.append(f"{enriched['junction'].nunique():,} junctions tracked")
    bits.append(f"{len(enriched):,} logged events")
    st.markdown('<div class="meta-row">' + "  ·  ".join(bits) + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 1) PREDICT EVENT
# --------------------------------------------------------------------------- #
if page == "Predict Event":
    with st.container(border=True):
        card_header("list", "Incident", "The signals the model actually weighs")
        c1, c2 = st.columns(2)
        with c1:
            cause = st.selectbox("Event cause", options(enriched, "event_cause") or ["Accident"])
            veh = st.selectbox("Vehicle type", [NOT_SET] + (options(enriched, "veh_type") or ["Car"]))
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], index=2)
        with c2:
            etype = st.selectbox("Type", ["unplanned", "planned"])
            when = st.time_input("Time of day", dt.time(18, 0))
            ps = st.selectbox("Police station", options(enriched, "police_station") or ["Unknown"])

        with st.expander("Location context (optional — feeds the map & location history)"):
            junction = st.selectbox("Junction", [NOT_SET] + (options(enriched, "junction") or ["Unknown"]))

        predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        predictor = load_predictor()
        junction_val = val(junction)
        event = {
            "start_datetime": f"2024-06-01 {when.strftime('%H:%M')}",
            "priority": priority, "event_type": etype, "event_cause": cause,
            "veh_type": val(veh), "police_station": ps, "junction": junction_val,
        }
        r = predictor.predict(event)

        conf = load_conformal()
        if conf and "closure_prob_raw" in r:
            import conformal as _cf
            label = _cf.classify(r["closure_prob_raw"], conf["qhat"])
            _tone = theme["green"] if label.startswith("CONFIDENT") else "#f59e0b"
            st.markdown(f'<span style="background:{_tone};color:#fff;font-weight:700;'
                        f'padding:3px 10px;border-radius:5px;font-size:0.85rem;">{label}</span> '
                        f'<span style="color:#94a3b8;font-size:0.8rem;">&nbsp;90%-coverage guaranteed</span>',
                        unsafe_allow_html=True)
            
        g1, g2 = st.columns([1, 1])
        with g1:
            risk_meter(r["closure_prob"])
        with g2:
            impact_tone = "bad" if r["impact_score"] >= 7 else "warn" if r["impact_score"] >= 4 else "good"
            stat_grid([
                stat_card("gauge", "Impact score", f"{r['impact_score']} / 10", r["impact_tier"], tone=impact_tone),
                stat_card("clock", "Expected clearance", f"{r['expected_clearance_mins']} min", "historical average", tone="info"),
            ])
            if not r["is_known_location"]:
                alert("warn", "New location", "No history for this junction — estimate uses zone/global fallback.")

        res = r["resources"]
        st.markdown(card_shell("shield", "Recommended response", "Barricading & staffing plan for this event", ""),
                    unsafe_allow_html=True)
        stat_grid([
            stat_card("octagon", "Barricade?", "Yes" if res["barricading_recommended"] else "No",
                      tone="bad" if res["barricading_recommended"] else "good"),
            stat_card("route", "Barricades", str(res["barricades"]), tone="info"),
            stat_card("users", "Personnel", str(res["personnel"]), tone="info"),
            stat_card("shield", "Supervisors", str(res["supervisors"]), tone="info"),
            stat_card("alert_triangle", "Rapid response", "Yes" if res["rapid_response_required"] else "No",
                      tone="bad" if res["rapid_response_required"] else "good"),
        ])

        if r.get("analogs"):
            ana = r["analogs"]
            with st.container(border=True):
                card_header("clock", "Grounded in historical analogs",
                            f"{ana['n_matched']} most-similar past incidents · "
                            f"clearance {ana['clearance_p25']}–{ana['clearance_p75']} min · "
                            f"{ana['analog_closure_rate']*100:.0f}% needed a closure")
                ax = pd.DataFrame(ana["examples"]).rename(columns={
                    "event_cause": "Cause", "veh_type": "Vehicle", "zone": "Zone",
                    "clearance_mins": "Clearance (min)", "needed_closure": "Needed closure"})
                st.dataframe(ax, use_container_width=True, hide_index=True)
                
        ex = pd.DataFrame(r["explanations"])
        if not ex.empty:
            with st.container(border=True):
                card_header("bar_chart", "Why this prediction", "Feature contributions to closure risk")
                ex["signed"] = ex["contribution"]
                fig = px.bar(ex.iloc[::-1], x="signed", y="feature", orientation="h",
                             color="signed", color_continuous_scale=[theme["primary"], theme["surface-alt"], theme["red"]],
                             color_continuous_midpoint=0,
                             labels={"signed": "contribution to closure risk"})
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(style_fig(fig, height=300), use_container_width=True)

        if enriched is not None and {"latitude", "longitude"}.issubset(enriched.columns):
            jrows = enriched[enriched["junction"].astype(str) == str(junction_val)]
            if not jrows.empty:
                with st.container(border=True):
                    card_header("map_pin", "Junction location", None)
                    st.map(pd.DataFrame({"lat": [jrows["latitude"].mean()],
                                         "lon": [jrows["longitude"].mean()]}), zoom=12)


# --------------------------------------------------------------------------- #
# 2) RISK MAP  (hotspots)
# --------------------------------------------------------------------------- #
elif page == "Risk Map":
    zh = load_csv("hotspot_zone_hour.csv")
    jl = load_csv("hotspot_junctions.csv")
    if zh is None or jl is None:
        alert("warn", "Hotspots not generated",
              "Run <code>python hotspots.py --data data/flipkart_gridlock.csv</code> to generate them.")
        st.stop()

    with st.container(border=True):
        card_header("grid", "Event load by zone × hour of day", None)
        zcol = zh.columns[0]
        mat = zh.set_index(zcol)
        fig = px.imshow(mat, aspect="auto",
                        color_continuous_scale=[theme["surface-alt"], theme["amber-soft"], theme["amber"], theme["red"]],
                        labels=dict(x="Hour of day", y="Zone", color="Events"))
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    render_leaderboard(jl)


# --------------------------------------------------------------------------- #
# 3a) ANALYTICS
# --------------------------------------------------------------------------- #
elif page == "Analytics":
    hourly = load_csv("hotspot_hourly.csv")
    dow = load_csv("hotspot_dow.csv")

    a, b = st.columns(2)
    with a:
        if hourly is not None:
            with st.container(border=True):
                card_header("bar_chart", "Events by hour of day", None)
                fig = px.bar(hourly, x="hour", y="events", color_discrete_sequence=[theme["primary"]])
                st.plotly_chart(style_fig(fig, height=280), use_container_width=True)
    with b:
        if dow is not None:
            with st.container(border=True):
                card_header("bar_chart", "Events by day of week", None)
                names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                dd = dow.copy(); dd["day"] = dd["day_of_week"].map(lambda i: names[int(i)])
                fig = px.bar(dd, x="day", y="events", color_discrete_sequence=[theme["primary"]])
                st.plotly_chart(style_fig(fig, height=280), use_container_width=True)

    if enriched is not None:
        c, d = st.columns(2)
        with c:
            if "event_cause" in enriched.columns:
                with st.container(border=True):
                    card_header("list", "Events by cause", None)
                    vc = enriched["event_cause"].value_counts().head(12).reset_index()
                    vc.columns = ["event_cause", "count"]
                    fig = px.bar(vc, x="count", y="event_cause", orientation="h",
                                 color_discrete_sequence=[theme["primary"]])
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)
        with d:
            if {"event_cause", "closure_int"}.issubset(enriched.columns):
                with st.container(border=True):
                    card_header("alert_triangle", "Closure rate by cause (%)", None)
                    cr = (enriched.groupby("event_cause")["closure_int"].mean()
                          .sort_values(ascending=False).head(12).reset_index())
                    cr["closure_int"] *= 100
                    fig = px.bar(cr, x="closure_int", y="event_cause", orientation="h",
                                 color="closure_int",
                                 color_continuous_scale=[theme["amber-soft"], theme["amber"], theme["red"]])
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    fig.update_coloraxes(showscale=False)
                    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

        if "clearance_mins" in enriched.columns:
            with st.container(border=True):
                card_header("clock", "Clearance time", "Descriptive — not predicted")
                cl = enriched[enriched["clearance_mins"].notna()]
                fig = px.histogram(cl, x="clearance_mins", nbins=40, color_discrete_sequence=[theme["primary"]])
                st.plotly_chart(style_fig(fig, height=280), use_container_width=True)

        sc = load_survival_curves(); sp = load_survival_params()

    if sc is not None:
        with st.container(border=True):
            card_header("activity", "Clearance survival analysis",
                        "P(incident still blocking the road) over time — handles censored long incidents")
            if sp:
                stat_grid([
                    stat_card("loop", "Median time-to-clear", f"{sp['medians'].get('overall', 0):.0f} min", tone="info"),
                    stat_card("check_circle", "Events used", f"{sp['n_observed'] + sp['n_censored']}",
                              f"{sp['n_censored']} censored (regression dropped these)", tone="good"),
                    stat_card("bar_chart", "Ranking concordance", f"{sp['concordance']:.2f}", "0.5 = random", tone="info"),
                ])
            plot = sc[sc["t"] <= 300]
            fig = px.line(plot, x="t", y="S", color="group")
            fig = style_fig(fig, height=340)
            fig.update_layout(xaxis_title="minutes since incident", yaxis_title="P(still blocking)")
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# 3b) EVENT CASCADES
# --------------------------------------------------------------------------- #
elif page == "Event Cascades":
    cas = load_cascades()
    if cas is None or len(cas) == 0:
        alert("warn", "No cascades available",
              "Run <code>python cascade.py --data data/flipkart_gridlock.csv</code> to generate them.")
        st.stop()
    
    hm = load_hawkes()
    n = hm.branching_factor; hl = hm.half_life_min; f60 = hm.expected_followons(60)
    stat_grid([
        stat_card("alert_triangle", "Branching factor", f"{n:.2f}",
                  "follow-on incidents per incident (Hawkes)", tone="bad" if n > 0.3 else "info"),
        stat_card("loop", "Cascade half-life", f"{hl:.0f} min", "elevated-risk window", tone="info"),
        stat_card("activity", "Expected follow-ons / 1h", f"{f60:.2f}", "after an incident", tone="info"),
    ])
    with st.container(border=True):
        card_header("activity", "Cascade risk after an incident",
                    "Self-exciting (Hawkes) intensity — spikes when an incident is reported, then decays")
        cda = st.columns(2)
        with cda[0]:
            n_recent = st.number_input("Incidents just reported here", 1, 5, 1)
        with cda[1]:
            horizon = st.slider("Minutes ahead", 30, 240, 180, 30)
        xs, ys = hm.decay_curve(n_recent=n_recent, horizon_min=horizon)
        dfc = pd.DataFrame({"minutes after incident": xs, "risk vs normal (x)": ys})
        fig = px.line(dfc, x="minutes after incident", y="risk vs normal (x)", markers=True,
                      color_discrete_sequence=[theme["red"]])
        fig = style_fig(fig, height=300)
        fig.add_hline(y=1.0, line_dash="dash", line_color=theme["muted"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Risk jumps to {ys[0]:.1f}× normal right after an incident, halving every {hl:.0f} min.")
        
    stat_grid([
        stat_card("alert_triangle", "Cascade pairs", f"{len(cas):,}", "overlapping incidents", tone="info"),
        stat_card("octagon", "High-risk (≥7)", f"{int((cas['cascade_risk'] >= 7).sum()):,}", "compound risk", tone="bad"),
        stat_card("map_pin", "Locations affected", f"{cas['location'].nunique():,}", "corridors + clusters", tone="neutral"),
    ])

    view = st.radio("Show", ["Real corridors", "Spatial clusters", "All"], horizontal=True, index=0)
    if view == "Real corridors":
        show = cas[cas["group_type"] == "corridor"]
    elif view == "Spatial clusters":
        show = cas[cas["group_type"] == "geo"]
    else:
        show = cas
    if len(show) == 0:
        alert("info", "Nothing to show", "No cascades of this type in the data.")
        st.stop()

    top = (show.groupby("location")
              .agg(pairs=("cascade_risk", "size"), avg_risk=("cascade_risk", "mean"))
              .sort_values("pairs", ascending=False).head(12).reset_index())

    with st.container(border=True):
        card_header("bar_chart", "Most cascade-prone locations", "Concurrent overlapping incidents per location")
        fig = px.bar(top.iloc[::-1], x="pairs", y="location", orientation="h", color="avg_risk",
                     color_continuous_scale=[theme["amber-soft"], theme["amber"], theme["red"]])
        fig = style_fig(fig, height=380)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        card_header("list", "Worst cascades", "Highest compound-risk overlaps")
        worst = show.sort_values("cascade_risk", ascending=False).head(15).copy()
        worst["window"] = (pd.to_datetime(worst["overlap_start"]).dt.strftime("%d %b %H:%M")
                           + " → " + pd.to_datetime(worst["overlap_end"]).dt.strftime("%H:%M"))
        st.dataframe(worst[["location", "window", "overlap_min", "closures_in_pair", "cascade_risk"]],
                     use_container_width=True, hide_index=True)

    focus = top.iloc[0]["location"]
    sub = show[show["location"] == focus].sort_values("cascade_risk", ascending=False).head(30)
    if len(sub):
        with st.container(border=True):
            card_header("clock", "Cascade timeline", f"Most-affected location · {focus}")
            tl = pd.DataFrame([{"Pair": f"{r.event_a}+{r.event_b}", "Start": r.overlap_start,
                                "End": r.overlap_end, "Risk": r.cascade_risk} for r in sub.itertuples()])
            fig = px.timeline(tl, x_start="Start", x_end="End", y="Pair", color="Risk",
                              color_continuous_scale=[theme["amber-soft"], theme["amber"], theme["red"]])
            fig = style_fig(fig, height=420)
            fig.update_yaxes(autorange="reversed")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 4) DIVERSION & BARRICADES  (routing page with theme support)
# --------------------------------------------------------------------------- #
elif page == "Diversion & Barricades":
    try:
        import routing_page
        # Call render_routing_page with theme and dark mode info
        routing_page.render_routing_page(theme=theme)
    except Exception as e:
        alert("bad", "Routing page failed to load", str(e))

# --------------------------------------------------------------------------- #
# 5) DEPLOYMENT OPTIMIZER
# --------------------------------------------------------------------------- #

elif page == "Deployment Optimizer":
    import optimizer

    mode = st.radio("Scenario", ["Simulated day", "Replay a real day"], horizontal=True)
    if mode == "Replay a real day":
        days = optimizer.available_days(top=20)
        pick = st.selectbox("Date", [f"{d} ({c} incidents)" for d, c in days])
        date_str = pick.split(" ")[0]
        pool, n_events = optimizer.load_real_day(date_str)
    else:
        n_events = st.slider("Predicted incidents (busy day ≈ 71)", 20, 250, 71, 10)
        pool = load_event_pool(n_events, 0)

    cols = st.columns(2)
    with cols[0]:
        officer_budget = st.slider("Officers available", 20, 300, 80, 10)
    with cols[1]:
        barricade_budget = st.slider("Barricades available", 5, 120, 30, 5)

    results = optimizer.compare(pool, officer_budget, barricade_budget)
    ilp = results[0]

    demand_o = int(pool["personnel"].sum())
    stat_grid([
        stat_card("users", "Officer demand", f"{demand_o}", f"budget {officer_budget}",
                  tone="bad" if demand_o > officer_budget else "good"),
        stat_card("octagon", "Events covered", f"{ilp['events_covered']}/{ilp['events_total']}",
                  "by optimizer", tone="info"),
        stat_card("shield", "Disruption mitigated", f"{ilp['coverage_pct']}%",
                  "of total weighted risk", tone="good"),
        stat_card("route", "Barricades used", f"{ilp['barricades_used']}/{barricade_budget}", tone="info"),
    ])

    with st.container(border=True):
        card_header("bar_chart", "Optimizer vs greedy baselines", "Weighted disruption mitigated for the SAME budget")
        comp = pd.DataFrame([{"method": r["label"], "importance": r["importance_captured"]} for r in results])
        fig = px.bar(comp, x="importance", y="method", orientation="h", color="importance",
                     color_continuous_scale=[theme["amber-soft"], theme["green"]])
        fig = style_fig(fig, height=240)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        baselines_best = max(r["importance_captured"] for r in results[1:])
        lift = ilp["importance_captured"] - baselines_best
        if lift > 0:
            alert("good", "Optimal allocation",
                  f"The ILP optimizer mitigates {lift:.1f} more weighted disruption than the best "
                  f"greedy approach for the same officers and barricades.")

    with st.container(border=True):
        card_header("list", "Selected deployment plan", "Events the optimizer chose to fully resource")
        sel = pool[ilp["selection"] == 1].sort_values("importance", ascending=False)
        show = sel[["event_cause", "closure_prob", "impact", "expected_clearance",
                    "personnel", "barricades", "importance"]].copy()
        show.columns = ["Cause", "Closure prob", "Impact", "Clearance (min)",
                        "Officers", "Barricades", "Priority weight"]
        st.dataframe(show, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# 6) LEARNING LOOP
# --------------------------------------------------------------------------- #
elif page == "Learning Loop":
    import time
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    import learning_loop as ll

    enr = load_enriched()

    # ---- Section 1: human-in-the-loop feedback (the deployment input mechanism) ----
    with st.container(border=True):
        card_header("loop", "Human-in-the-loop feedback",
                    "In real deployment, a dispatcher confirms each outcome with one tap")
        if "fb_event" not in st.session_state:
            st.session_state.fb_event = enr.sample(1).iloc[0].to_dict()
        ev = st.session_state.fb_event
        prob = float(ev.get("pred_closure_prob", 0.1) or 0.1)
        st.markdown(f"**Incident:** {ev.get('event_cause','?')} · {ev.get('veh_type','?')} "
                    f"&nbsp;|&nbsp; model predicted closure risk **{prob*100:.0f}%**")
        cc = st.columns(3)
        if cc[0].button("✅ Closure WAS needed", use_container_width=True):
            ll.log_outcome(ev.get("id", "manual"), prob, 1)
            st.session_state.fb_event = enr.sample(1).iloc[0].to_dict(); st.toast("Logged ✓"); st.rerun()
        if cc[1].button("❌ NOT needed", use_container_width=True):
            ll.log_outcome(ev.get("id", "manual"), prob, 0)
            st.session_state.fb_event = enr.sample(1).iloc[0].to_dict(); st.toast("Logged ✓"); st.rerun()
        if cc[2].button("🔄 Recalibrate now", use_container_width=True):
            n = ll.recalibrate_from_feedback(min_n=1)
            st.success(f"Recalibrated from {n} logged outcomes." if n else "Need more varied feedback first.")
        sfb = ll.feedback_stats()
        st.caption(f"Outcomes logged this session: {sfb['n']}. In production these accumulate and recalibrate "
                   "the model continuously — the curves below prove that doing so improves accuracy.")

    # ---- Section 2: live proof the feedback improves the model ----
    with st.container(border=True):
        card_header("bar_chart", "Proof it works: watch calibration error drop as feedback streams in",
                    "Real outcomes replayed chronologically — static model vs continuous relearning")
        colA, colB = st.columns([1, 1])
        with colA:
            speed = st.select_slider("Speed", ["slow", "medium", "fast"], value="medium")
        with colB:
            start = st.button("▶ Start live feedback", type="primary", use_container_width=True)
        metric_ph = st.empty(); chart_ph = st.empty(); status_ph = st.empty()

        p, y = ll._raw_probs_chrono("data/flipkart_gridlock.csv")
        n = len(p); step = max(1, n // 40)
        pause = {"slow": 0.25, "medium": 0.10, "fast": 0.03}[speed]

        def _ece(yt, pt, bins=10):
            edges = np.linspace(0, 1, bins + 1)
            idx = np.clip(np.digitize(pt, edges) - 1, 0, bins - 1)
            e = 0.0
            for b in range(bins):
                m = idx == b
                if m.sum():
                    e += (m.sum() / len(yt)) * abs(pt[m].mean() - yt[m].mean())
            return e

        if start:
            acc_p, acc_y, history = [], [], []
            cal = None
            for i in range(0, n, step):
                bp, by = p[i:i + step], y[i:i + step]
                if len(bp) == 0:
                    continue
                cp = cal.predict(bp) if cal is not None else bp
                if len(set(by)) > 1:
                    history.append({"outcomes_seen": i + len(bp),
                                    "static (no learning)": round(_ece(by, bp), 4),
                                    "learning": round(_ece(by, cp), 4)})
                acc_p.extend(bp.tolist()); acc_y.extend(by.tolist())
                if len(set(acc_y)) > 1:
                    cal = IsotonicRegression(out_of_bounds="clip").fit(np.array(acc_p), np.array(acc_y))
                if len(history) >= 2:
                    warm = pd.DataFrame(history).iloc[1:]
                    ms, ml = warm["static (no learning)"].mean(), warm["learning"].mean()
                    drop = (100 * (ms - ml) / ms) if ms else 0
                    metric_ph.metric("Avg calibration error — with learning", f"{ml:.3f}",
                                     f"{drop:.0f}% lower than static", delta_color="inverse")
                    dfh = pd.DataFrame(history).melt(id_vars="outcomes_seen",
                            value_vars=["static (no learning)", "learning"],
                            var_name="mode", value_name="ECE")
                    fig = px.line(dfh, x="outcomes_seen", y="ECE", color="mode", markers=True,
                                  color_discrete_map={"static (no learning)": theme["red"],
                                                      "learning": theme["green"]})
                    fig = style_fig(fig, height=320)
                    chart_ph.plotly_chart(fig, use_container_width=True)
                status_ph.caption(f"Fed {i + len(bp):,} of {n:,} real outcomes")
                time.sleep(pause)
            warm = pd.DataFrame(history).iloc[1:]



# --------------------------------------------------------------------------- #
# 7) LIVE FEED (real TomTom feed if key set, else replay a real day live)
# --------------------------------------------------------------------------- #
elif page == "Live Feed":
    import time, live_feed
    hm = load_hawkes()

    # ---- Mode A: real live incidents from TomTom ----
    if live_feed.source_status() == "live-api":
        st.markdown('<p style="color:var(--ink);font-weight:600;">🟢 Live — TomTom incident feed, Bengaluru</p>',
                    unsafe_allow_html=True)
        if st.button("🔄 Refresh live incidents", type="primary"):
            st.rerun()
        live = live_feed.live_scored(load_predictor())
        if live is None or live.empty:
            st.info("No live incidents reported in Bengaluru right now. "
                    "(Try during peak hours — the replay mode below always has data.)")
        else:
            risk = live_feed.live_cascade_risk(live, 0, hm)
            stat_grid([
                stat_card("activity", "Live incidents now", f"{len(live)}", "TomTom feed", tone="info"),
                stat_card("alert_triangle", "Expected follow-ons (next 1h)", f"{risk:.1f}",
                          "from live incidents (Hawkes)", tone="bad" if risk > 3 else "info"),
                stat_card("shield", "Need pre-position",
                          f"{int((live['readiness'] == 'PRE-POSITION').sum())}", "high-risk now", tone="bad"),
            ])
            mapfig = live_feed.risk_map_figure(live, hm)
            if mapfig is not None:
                lc, rc = st.columns([1, 1])
                with lc:
                    show = live[["time", "cause", "location", "closure_prob", "impact", "readiness"]].copy()
                    show.columns = ["Time", "Cause", "Location", "Closure prob", "Impact", "Readiness"]
                    st.dataframe(show, use_container_width=True, hide_index=True, height=440)
                with rc:
                    st.plotly_chart(mapfig, use_container_width=True)
            else:
                show = live[["time", "cause", "location", "closure_prob", "impact", "readiness"]].copy()
                show.columns = ["Time", "Cause", "Location", "Closure prob", "Impact", "Readiness"]
                st.dataframe(show, use_container_width=True, hide_index=True)

        

    # ---- Mode B: replay a real high-incident day as a live stream ----
    else:
        df, day = load_replay()
        src_note = "replaying real incidents from " + day
        st.caption(f"▶ {src_note} · {len(df)} incidents")
        cc = st.columns([1, 1])
        with cc[0]:
            speed = st.select_slider("Speed", ["slow", "medium", "fast"], value="medium")
        with cc[1]:
            play = st.button("▶ Start live feed", type="primary", use_container_width=True)
        metric_ph = st.empty(); table_ph = st.empty(); status_ph = st.empty()
        pause = {"slow": 0.30, "medium": 0.15, "fast": 0.05}[speed]
        tmax = df["t_min"].max(); step = max(tmax / 80.0, 1.0)

        if play:
            sim = 0.0
            while sim <= tmax:
                act = live_feed.active_incidents(df, sim, lookback_min=45)
                risk = live_feed.live_cascade_risk(act, sim, hm)
                high = int((act["readiness"] == "PRE-POSITION").sum()) if len(act) else 0
                with metric_ph.container():
                    stat_grid([
                        stat_card("activity", "Active incidents", f"{len(act)}",
                                  f"{int(sim//60):02d}h{int(sim%60):02d}m into the day", tone="info"),
                        stat_card("alert_triangle", "Expected follow-ons (next 1h)", f"{risk:.1f}",
                              "from active incidents (Hawkes)", tone="bad" if risk > 3 else "info"),
                        stat_card("shield", "Need pre-position", f"{high}",
                                  "high-risk active now", tone="bad" if high else "good"),
                    ])
                with table_ph.container():
                    if len(act):
                        show = act[["time", "cause", "location", "closure_prob", "impact", "readiness"]].copy()
                        show.columns = ["Time", "Cause", "Location", "Closure prob", "Impact", "Readiness"]
                        st.dataframe(show, use_container_width=True, hide_index=True)
                    else:
                        st.info("No active incidents in this window yet…")
                status_ph.caption(f"Simulated time: {int(sim//60):02d}:{int(sim%60):02d}")
                sim += step; time.sleep(pause)
            status_ph.success(f"Replay complete — {len(df)} real incidents from {day} streamed and scored live.")
        else:
            st.info("Press **Start live feed** to stream a real high-incident day, scored live by the system.")


            