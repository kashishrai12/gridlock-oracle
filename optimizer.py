"""
optimizer.py — City-wide deployment optimizer (operations research).

Per-event recommendations are easy. The real problem is ALLOCATION UNDER SCARCITY:
given a day of predicted incidents and a LIMITED pool of officers + barricades, which
events do we fully resource to mitigate the most expected disruption?

With two resource budgets this is a multi-dimensional 0/1 knapsack — NP-hard, and
greedy is provably sub-optimal — so we solve it exactly with integer programming
(scipy HiGHS). We also run greedy baselines to quantify the optimizer's lift.

Build pool from predictions, then:
  python optimizer.py --officers 80 --barricades 30
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

from analogs import AnalogRetriever

ENRICHED = "models/enriched_dataset.csv"

# Measured from the dataset: incidents per day (ground the scenario, don't guess).
EVENTS_PER_DAY_MEDIAN = 49     # typical day
EVENTS_PER_DAY_P90 = 71        # busy day
EVENTS_PER_DAY_MAX = 250       # worst day on record


# --------------------------------------------------------------------------- #
# Build the day's event pool from existing predictions
# --------------------------------------------------------------------------- #
def _attach_needs(samp):
    """Attach per-event closure prob, impact, resource needs and importance."""
    prob = pd.to_numeric(samp.get("pred_closure_prob", 0.1), errors="coerce").fillna(0.1).values
    impact = pd.to_numeric(samp.get("impact_score", 3.0), errors="coerce").fillna(3.0).values
    clr = pd.to_numeric(samp["clearance_mins"], errors="coerce").fillna(60).clip(upper=480).values

    personnel, barricades = [], []
    for c, p in zip(clr, prob):
        rp = AnalogRetriever.resource_plan(int(c), float(p))
        personnel.append(rp["personnel"])
        barricades.append(rp["barricades"])

    pool = pd.DataFrame({
        "event_id": samp.get("id", pd.Series(range(len(samp)))).values,
        "event_cause": samp.get("event_cause", "").astype(str).values,
        "closure_prob": np.round(prob, 3),
        "impact": np.round(impact, 2),
        "expected_clearance": clr.astype(int),
        "personnel": personnel,
        "barricades": barricades,
    })
    pool["importance"] = np.round(pool["impact"] * (0.5 + pool["closure_prob"]), 2)
    return pool


def build_event_pool(n=EVENTS_PER_DAY_P90, seed=0, source=ENRICHED):
    """Sample n historical incidents as a 'what-if' day. Default = a busy (p90) day,
    measured from the data, not guessed."""
    df = pd.read_csv(source)
    df = df[df["clearance_mins"].notna()].copy()
    samp = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    return _attach_needs(samp)


def load_real_day(date_str, source=ENRICHED):
    """Load the ACTUAL incidents that occurred on a real date (e.g. '2024-03-14').
    Returns (pool, n_events). This is the strongest framing: optimise a real day."""
    df = pd.read_csv(source)
    df = df[df["clearance_mins"].notna()].copy()
    if "start_datetime" not in df.columns:
        return build_event_pool(), 0
    day = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.date.astype(str)
    samp = df[day == str(date_str)].reset_index(drop=True)
    if len(samp) == 0:
        return build_event_pool(), 0
    return _attach_needs(samp), len(samp)


def available_days(source=ENRICHED, top=30):
    """Busiest real days in the data, for a 'replay this day' picker."""
    df = pd.read_csv(source)
    df = df[df["clearance_mins"].notna()].copy()
    if "start_datetime" not in df.columns:
        return []
    day = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.date.astype(str)
    vc = day.value_counts().head(top)
    return [(d, int(c)) for d, c in vc.items()]


# --------------------------------------------------------------------------- #
# Summary helper
# --------------------------------------------------------------------------- #
def _summarize(pool, x, officer_budget, barricade_budget, label):
    x = np.asarray(x).astype(int)
    sel = pool[x == 1]
    total_w = pool["importance"].sum()
    return {
        "label": label,
        "events_covered": int(x.sum()),
        "events_total": len(pool),
        "importance_captured": round(float(sel["importance"].sum()), 1),
        "importance_total": round(float(total_w), 1),
        "coverage_pct": round(100 * sel["importance"].sum() / total_w, 1) if total_w else 0.0,
        "officers_used": int(sel["personnel"].sum()),
        "officer_budget": officer_budget,
        "barricades_used": int(sel["barricades"].sum()),
        "barricade_budget": barricade_budget,
        "selection": x,
    }


# --------------------------------------------------------------------------- #
# ILP optimizer (exact) — maximise mitigated disruption under two budgets
# --------------------------------------------------------------------------- #
def optimize_deployment(pool, officer_budget, barricade_budget):
    w = pool["importance"].values.astype(float)
    p = pool["personnel"].values.astype(float)
    b = pool["barricades"].values.astype(float)
    n = len(pool)

    try:
        res = milp(
            c=-w,                                       # milp minimises -> negate to maximise
            constraints=[LinearConstraint(p, ub=officer_budget),
                         LinearConstraint(b, ub=barricade_budget)],
            integrality=np.ones(n),
            bounds=Bounds(0, 1),
        )
        x = np.round(res.x).astype(int) if (res.success and res.x is not None) else _greedy(pool, officer_budget, barricade_budget, "ratio")
        label = "ILP optimizer (HiGHS)" if res.success else "greedy fallback"
    except Exception:
        x = _greedy(pool, officer_budget, barricade_budget, "ratio")
        label = "greedy fallback"
    return _summarize(pool, x, officer_budget, barricade_budget, label)


# --------------------------------------------------------------------------- #
# Greedy baselines (what teams usually do) — for the comparison
# --------------------------------------------------------------------------- #
def _greedy(pool, officer_budget, barricade_budget, key):
    if key == "ratio":
        score = pool["importance"] / pool["personnel"].clip(lower=1)
    elif key == "impact":
        score = pool["impact"]
    else:
        score = pool["importance"]
    order = score.sort_values(ascending=False).index
    x = np.zeros(len(pool), int)
    pos = {idx: i for i, idx in enumerate(pool.index)}
    used_o = used_b = 0
    for idx in order:
        i = pos[idx]
        pe, ba = pool["personnel"].iloc[i], pool["barricades"].iloc[i]
        if used_o + pe <= officer_budget and used_b + ba <= barricade_budget:
            x[i] = 1; used_o += pe; used_b += ba
    return x


def greedy_deployment(pool, officer_budget, barricade_budget, key="importance"):
    x = _greedy(pool, officer_budget, barricade_budget, key)
    names = {"importance": "Greedy by importance", "impact": "Greedy by impact",
             "ratio": "Greedy by importance/officer"}
    return _summarize(pool, x, officer_budget, barricade_budget, names.get(key, key))


def compare(pool, officer_budget, barricade_budget):
    return [
        optimize_deployment(pool, officer_budget, barricade_budget),
        greedy_deployment(pool, officer_budget, barricade_budget, "impact"),
        greedy_deployment(pool, officer_budget, barricade_budget, "importance"),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=EVENTS_PER_DAY_P90)
    ap.add_argument("--officers", type=int, default=80)
    ap.add_argument("--barricades", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = build_event_pool(n=a.n, seed=a.seed)
    print(f"\n[pool] {len(pool)} events | total officer demand={int(pool['personnel'].sum())} "
          f"barricade demand={int(pool['barricades'].sum())}")
    print(f"[budget] officers={a.officers} barricades={a.barricades}\n")
    print(f"{'method':<28}{'covered':>9}{'importance':>13}{'cov%':>7}{'officers':>10}{'barr':>6}")
    for r in compare(pool, a.officers, a.barricades):
        print(f"{r['label']:<28}{r['events_covered']:>4}/{r['events_total']:<4}"
              f"{r['importance_captured']:>13}{r['coverage_pct']:>6}%"
              f"{r['officers_used']:>7}/{r['officer_budget']:<2}{r['barricades_used']:>4}")
    print()