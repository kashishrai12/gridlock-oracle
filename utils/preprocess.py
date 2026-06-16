"""
utils/preprocess.py  —  Day-1 rewrite

What changed vs the old version and WHY:
  1. TARGET is now real, observed CLEARANCE TIME (resolved_datetime - start_datetime),
     not the hand-built Disruption Score. Kills the "you trained a model to copy your
     own formula" critique.
  2. We train on log(clearance) because clearance is heavily right-skewed.
  3. LEAKAGE STRIPPED: no feature is derived from end_datetime / resolved_datetime /
     status / duration. Those are only known AFTER the event ends.
  4. Location aggregates (junction / corridor / zone) are FIT ON TRAIN ONLY and applied
     with a junction -> zone -> global fallback chain (cold-start safe).
  5. Disruption Score still exists, but is now a TRANSPARENT function of the *predicted*
     clearance time (see derive_disruption_score). It is an output layer, not the label.

Public API used by train_model.py / predictor.py:
  - load_and_prepare(path)                  -> clean df with target + base features
  - temporal_split(df, train_frac)          -> (train_df, test_df) ordered by time
  - fit_location_stats(train_df)            -> stats dict (train-only)
  - apply_location_stats(df, stats)         -> df with location features + flags
  - FEATURE_COLS                            -> the exact, leakage-free input columns
  - derive_disruption_score(...)            -> deterministic 0-10 business tier
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #

# Columns that encode the FUTURE (only known once the event is over). Never inputs.
LEAKAGE_COLS = ["end_datetime", "resolved_datetime", "status", "duration_mins",
                "resolved", "disruption_score"]

# The leakage-free feature set the model is allowed to see. start_datetime-derived
# time features + static attributes known at event-creation time + train-only aggregates.
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "priority_encoded", "is_planned", "closure_int",
    "junction_event_count", "junction_closure_rate", "junction_hist_clearance",
    "corridor_event_count", "corridor_closure_rate",
    "zone_event_count",
    "is_known_junction",
]

# Extra raw columns that plausibly drive clearance time and are known at event
# creation (no leakage). Added via XGBoost native categorical / numeric handling.
# Self-profiled at fit time: only included if non-null rate and cardinality are sane.
CATEGORICAL_CANDIDATES = ["event_cause", "veh_type", "cargo_material",
                          "reason_breakdown", "direction", "police_station"]
NUMERIC_CANDIDATES = ["age_of_truck"]
MIN_FILL_RATE = 0.40       # need >=40% non-null to include a column
MAX_CAT_CARDINALITY = 60   # skip ultra-high-cardinality cols (too sparse to learn)

TARGET_RAW = "clearance_mins"     # real observed minutes
TARGET = "log_clearance"          # what the model actually regresses on

# Reliable-clearance window. Durations outside this are treated as data artifacts
# (sub-minute = clock noise; multi-day = administrative batch-close, not real
# clearance) and excluded from training. Tune MAX after seeing your percentiles.
MIN_CLEARANCE_MINS = 1
MAX_CLEARANCE_MINS = 1440         # 24h. p90 in this data is ~10 days = admin close.

# Map common priority encodings to an ordinal scale. Robust to str or int inputs.
_PRIORITY_MAP = {
    "low": 0, "l": 0, "p4": 0, "4": 0, "1": 0,        # adjust to YOUR dataset's values
    "medium": 1, "med": 1, "m": 1, "p3": 1, "3": 1, "2": 1,
    "high": 2, "h": 2, "p2": 2,
    "critical": 3, "crit": 3, "c": 3, "p1": 3,
}


# ----------------------------------------------------------------------------- #
# Parsing / target
# ----------------------------------------------------------------------------- #

def _parse_datetimes(df):
    for c in ["start_datetime", "end_datetime", "resolved_datetime", "closed_datetime"]:
        if c in df.columns:
            # utc=True normalizes the '+00' offsets in this dataset and keeps dtype clean.
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def _compute_target(df):
    """
    Clearance time = (event end) - start_datetime, in minutes.

    "event end" is taken from the first available of:
        resolved_datetime -> closed_datetime -> end_datetime
    This matches the incident lifecycle: most events are terminally 'closed', so
    closed_datetime is the workhorse signal. Using these to BUILD THE LABEL is not
    leakage — they are never input features; we only use them to compute the
    ground-truth clearance time we predict.

    Records whose duration falls outside [MIN, MAX] are excluded: sub-minute values
    are clock noise, multi-day values are administrative batch-closes (this dataset's
    p90 is ~10 days) rather than real road-clearance times. Excluding them is a
    deliberate, stated data-quality decision — not cherry-picking.
    """
    def col(name):
        return df[name] if name in df.columns else pd.Series(pd.NaT, index=df.index)

    best_end = col("resolved_datetime").fillna(col("closed_datetime")).fillna(col("end_datetime"))
    best_end = pd.to_datetime(best_end, errors="coerce", utc=True)
    src = np.where(col("resolved_datetime").notna(), "resolved",
          np.where(col("closed_datetime").notna(), "closed", "end"))
    df["end_source"] = src

    delta = (best_end - df["start_datetime"]).dt.total_seconds() / 60.0
    df[TARGET_RAW] = delta

    before = len(df)
    has_dur = delta.notna()
    in_window = has_dur & (delta >= MIN_CLEARANCE_MINS) & (delta <= MAX_CLEARANCE_MINS)
    n_long = int((has_dur & (delta > MAX_CLEARANCE_MINS)).sum())
    n_none = int((~has_dur).sum())

    df = df[in_window].copy()
    by_src = pd.Series(df["end_source"]).value_counts().to_dict()
    print(f"[target] usable clearances: {len(df)}/{before}  "
          f"(excluded: {n_none} no end-signal, {n_long} >{MAX_CLEARANCE_MINS}min admin-close)")
    print(f"[target] source breakdown: {by_src}  | median={df[TARGET_RAW].median():.0f} min")

    df[TARGET] = np.log1p(df[TARGET_RAW])
    return df


# ----------------------------------------------------------------------------- #
# Leakage-free base features (all derivable at event-creation time)
# ----------------------------------------------------------------------------- #

def _time_features(df):
    s = df["start_datetime"]
    df["hour"] = s.dt.hour
    df["day_of_week"] = s.dt.dayofweek
    df["month"] = s.dt.month
    df["is_weekend"] = (s.dt.dayofweek >= 5).astype(int)
    return df


def _encode_priority(val):
    if pd.isna(val):
        return -1
    key = str(val).strip().lower()
    return _PRIORITY_MAP.get(key, -1)


def _static_features(df):
    # priority
    if "priority" in df.columns:
        df["priority_encoded"] = df["priority"].map(_encode_priority).astype(int)
    else:
        df["priority_encoded"] = -1

    # planned vs unplanned. This dataset's event_type is literally 'planned'/'unplanned'.
    # NB: must use exact match — .contains('planned') wrongly matches 'unplanned'.
    if "is_planned" in df.columns:
        df["is_planned"] = df["is_planned"].astype(int)
    elif "event_type" in df.columns:
        et = df["event_type"].astype(str).str.strip().str.lower()
        df["is_planned"] = (et == "planned").astype(int)
    else:
        df["is_planned"] = 0

    # road closure flag
    if "requires_road_closure" in df.columns:
        df["closure_int"] = (
            df["requires_road_closure"]
              .astype(str).str.lower()
              .isin(["1", "true", "yes", "y", "t"]).astype(int)
        )
    else:
        df["closure_int"] = 0
    return df


# ----------------------------------------------------------------------------- #
# Location aggregates — FIT ON TRAIN ONLY, applied with fallback chain
# ----------------------------------------------------------------------------- #

def fit_location_stats(train_df, smoothing=20):
    """
    Build lookup tables from TRAIN rows only. No test data touches these.
    Historical-clearance encodings are SHRUNK toward the global mean by count
    (Bayesian smoothing) so rare junctions don't memorise a single event.
    """
    g_clear = float(train_df[TARGET_RAW].mean())
    g_closure = float(train_df["closure_int"].mean())

    def agg(col):
        grp = train_df.groupby(col)
        out = grp.agg(
            count=("closure_int", "size"),
            closure_rate=("closure_int", "mean"),
            mean_clear=(TARGET_RAW, "mean"),
        )
        # shrink mean_clear toward global by count
        n = out["count"]
        out["mean_clear_smooth"] = (out["mean_clear"] * n + g_clear * smoothing) / (n + smoothing)
        return out

    stats = {
        "global": {"clear": g_clear, "closure": g_closure},
        "junction": agg("junction") if "junction" in train_df else None,
        "corridor": agg("corridor") if "corridor" in train_df else None,
        "zone": agg("zone") if "zone" in train_df else None,
    }

    # ---- Self-profile extra features on TRAIN only ----
    cat_levels, num_fill = {}, {}
    print("[features] profiling extra columns (train only):")
    for c in CATEGORICAL_CANDIDATES:
        if c not in train_df.columns:
            continue
        fill = train_df[c].notna().mean()
        card = train_df[c].nunique(dropna=True)
        keep = (fill >= MIN_FILL_RATE) and (card <= MAX_CAT_CARDINALITY) and (card >= 2)
        print(f"    {c:18s} fill={fill:5.0%}  card={card:<4}  -> {'KEEP' if keep else 'skip'}")
        if keep:
            # store the category vocabulary from train so test/inference align.
            # dedupe (value_counts index is unique, but guard) and ensure single UNK.
            levels = list(dict.fromkeys(train_df[c].astype(str).fillna("UNK").value_counts().index))
            if "UNK" not in levels:
                levels.append("UNK")
            cat_levels[c] = levels
    for c in NUMERIC_CANDIDATES:
        if c not in train_df.columns:
            continue
        vals = pd.to_numeric(train_df[c], errors="coerce")
        fill = vals.notna().mean()
        keep = fill >= MIN_FILL_RATE
        print(f"    {c:18s} fill={fill:5.0%}  (numeric)  -> {'KEEP' if keep else 'skip'}")
        if keep:
            num_fill[c] = float(vals.median())

    stats["cat_levels"] = cat_levels
    stats["num_fill"] = num_fill
    return stats


def apply_location_stats(df, stats):
    """
    Apply train-only stats with a junction -> zone -> global fallback so a never-seen
    location degrades gracefully instead of producing NaNs. is_known_junction is a
    feature so the model can learn to trust cold-start rows less.
    """
    df = df.copy()   # avoid SettingWithCopyWarning on slices from temporal_split
    g = stats["global"]

    def lookup(level, key_col, count_col, rate_col, clear_col, default_clear, default_rate):
        tbl = stats[level]
        if tbl is None or key_col not in df.columns:
            df[count_col] = 0
            df[rate_col] = default_rate
            if clear_col is not None:
                df[clear_col] = default_clear
            return
        df[count_col] = df[key_col].map(tbl["count"]).fillna(0).astype(float)
        df[rate_col] = df[key_col].map(tbl["closure_rate"]).fillna(default_rate)
        if clear_col is not None:
            df[clear_col] = df[key_col].map(tbl["mean_clear_smooth"]).fillna(default_clear)

    lookup("junction", "junction",
           "junction_event_count", "junction_closure_rate", "junction_hist_clearance",
           g["clear"], g["closure"])
    lookup("corridor", "corridor",
           "corridor_event_count", "corridor_closure_rate", None,
           g["clear"], g["closure"])
    lookup("zone", "zone",
           "zone_event_count", "zone_closure_rate_tmp", None,
           g["clear"], g["closure"])
    df.drop(columns=[c for c in ["zone_closure_rate_tmp"] if c in df], inplace=True)

    # cold-start flag
    if stats["junction"] is not None and "junction" in df.columns:
        known = set(stats["junction"].index)
        df["is_known_junction"] = df["junction"].isin(known).astype(int)
    else:
        df["is_known_junction"] = 0

    # ---- extra categorical features, aligned to TRAIN vocabulary ----
    for c, levels in stats.get("cat_levels", {}).items():
        raw = df[c].astype(str).fillna("UNK") if c in df.columns else pd.Series("UNK", index=df.index)
        raw = raw.where(raw.isin(levels), "UNK")          # unseen category -> UNK
        df[c + "_cat"] = pd.Categorical(raw, categories=levels)
    # ---- extra numeric features, filled with train median ----
    for c, fill in stats.get("num_fill", {}).items():
        v = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
        df[c + "_num"] = v.fillna(fill)
    return df


def get_feature_cols(stats):
    """Full feature list = leakage-free base + profiled extras. Returns (all_cols,
    categorical_cols) so the trainer can enable XGBoost native categorical."""
    cat_cols = [c + "_cat" for c in stats.get("cat_levels", {})]
    num_cols = [c + "_num" for c in stats.get("num_fill", {})]
    return FEATURE_COLS + num_cols + cat_cols, cat_cols


# ----------------------------------------------------------------------------- #
# Orchestration
# ----------------------------------------------------------------------------- #

def load_and_prepare(path):
    """Read CSV -> parse -> real target -> leakage-free base features. No aggregates yet
    (those are fit AFTER the temporal split to avoid cross-split leakage)."""
    df = pd.read_csv(path)
    df = _parse_datetimes(df)
    if "start_datetime" not in df.columns:
        raise ValueError("start_datetime required.")
    df = df[df["start_datetime"].notna()].copy()
    df = _compute_target(df)
    df = _time_features(df)
    df = _static_features(df)
    return df


def temporal_split(df, train_frac=0.8, time_col="start_datetime"):
    """Forecasting-honest split: oldest train_frac by time -> train, newest -> test.
    This is the change that makes the metrics mean something."""
    df = df.sort_values(time_col).reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def build_xy(df, stats):
    """Attach train-only location + extra features and return (X, y, df) ready for
    XGBoost. Categorical columns keep 'category' dtype for native handling."""
    df = apply_location_stats(df, stats)
    feat_cols, cat_cols = get_feature_cols(stats)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feat_cols].copy()
    for c in feat_cols:
        if c not in cat_cols:
            X[c] = X[c].astype(float)
    y = df[TARGET].astype(float)
    return X, y, df


# ----------------------------------------------------------------------------- #
# Disruption Score is now an OUTPUT, not a label. Transparent + defensible.
# ----------------------------------------------------------------------------- #

def derive_disruption_score(pred_clearance_mins, priority_encoded, closure_int,
                            junction_hist_clearance, global_clear):
    """
    Deterministic 0-10 score computed FROM the model's predicted clearance time.
    No black box: every term is explainable to a judge.
      - longer predicted clearance        -> higher
      - higher priority                   -> higher
      - road closure                      -> higher
      - junction historically bad         -> higher
    """
    # normalise predicted clearance against the global mean (≈1.0 = average event)
    rel = pred_clearance_mins / max(global_clear, 1.0)
    base = 4.0 * np.tanh(rel)                       # 0..~4, saturating
    base += 1.5 * (priority_encoded / 3.0)          # 0..1.5
    base += 1.5 * closure_int                       # 0 or 1.5
    base += 1.5 * np.tanh(junction_hist_clearance / max(global_clear, 1.0) - 1.0 + 1.0) \
            if junction_hist_clearance else 0.0
    score = float(np.clip(base + 1.0, 0, 10))       # +1 floor so trivial events aren't 0
    return round(score, 1)


def impact_tier(score):
    if score >= 8:   return "CRITICAL"
    if score >= 6:   return "HIGH"
    if score >= 3.5: return "MODERATE"
    return "LOW"


# ----------------------------------------------------------------------------- #
# Closure-classifier feature contract + single-event inference (used by predictor)
# ----------------------------------------------------------------------------- #

# closure-derived aggregates excluded from the classifier to stay leakage-clean
CLOSURE_EXCLUDE = {"closure_int", "junction_closure_rate", "corridor_closure_rate"}


def classifier_feature_cols(stats):
    """Feature contract for the closure classifier. Single source of truth shared by
    train_model.py and predictor.py so training and inference never diverge."""
    all_cols, cat_cols = get_feature_cols(stats)
    feats = [c for c in all_cols if c not in CLOSURE_EXCLUDE]
    cats = [c for c in cat_cols if c not in CLOSURE_EXCLUDE]
    return feats, cats


def closure_impact_score(p_closure, priority_encoded, is_planned,
                         junction_hist_clearance, global_clear):
    """Transparent 0-10 impact score anchored on the VALIDATED closure probability
    plus policy context. Not a learned label -> no circularity."""
    s = 5.5 * p_closure
    s += 1.5 * (priority_encoded / 3.0)
    s += 1.0 * is_planned
    if junction_hist_clearance:
        s += 2.0 * np.tanh(junction_hist_clearance / max(global_clear, 1.0))
    return round(float(np.clip(s, 0, 10)), 1)


def featurize_event(event: dict, stats):
    """Turn one raw event dict into a 1-row feature frame matching the classifier
    contract. Mirrors the training feature path exactly (time -> static -> location
    -> extras). Returns (X, enriched_row_df)."""
    df = pd.DataFrame([dict(event)])
    df = _parse_datetimes(df)
    if "start_datetime" not in df.columns or df["start_datetime"].isna().all():
        df["start_datetime"] = pd.Timestamp.now(tz="UTC")
    df = _time_features(df)
    df = _static_features(df)
    df = apply_location_stats(df, stats)
    feats, cats = classifier_feature_cols(stats)
    for c in feats:
        if c not in df.columns:
            df[c] = 0
    X = df[feats].copy()
    for c in feats:
        if c not in cats:
            X[c] = X[c].astype(float)
    return X, df