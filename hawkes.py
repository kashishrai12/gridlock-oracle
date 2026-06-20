

import argparse, math, pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODELS_DIR = "models"
GEO_CELL = 0.003
INVALID = {"", "nan", "none", "noncorridor", "non-corridor", "unknown", "na", "null"}


def _loc_key(df):
    corr = df["corridor"].astype(str).str.strip() if "corridor" in df.columns else pd.Series("", index=df.index)
    loc = corr.where(~corr.str.lower().isin(INVALID))
    if {"latitude", "longitude"}.issubset(df.columns):
        la = (pd.to_numeric(df["latitude"], errors="coerce") / GEO_CELL).round() * GEO_CELL
        lo = (pd.to_numeric(df["longitude"], errors="coerce") / GEO_CELL).round() * GEO_CELL
        loc = loc.fillna("geo " + la.round(3).astype(str) + "," + lo.round(3).astype(str))
    if "junction" in df.columns:
        j = df["junction"].astype(str).str.strip()
        loc = loc.fillna(j.where(~j.str.lower().isin(INVALID)))
    return loc


def _load_groups(path):
    """Return (list of per-location sorted event-time arrays in minutes, T_total_minutes)."""
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()]
    t0 = ts.min()
    df["t_min"] = (ts - t0).dt.total_seconds().values / 60.0
    df["loc"] = _loc_key(df)
    df = df[df["loc"].notna() & (df["loc"].astype(str) != "nan")]
    T = float(df["t_min"].max())
    groups = [np.sort(g["t_min"].values) for _, g in df.groupby("loc") if len(g) >= 1]
    return groups, T, len(df)


def _neg_ll(params, groups, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:   # alpha<beta keeps it subcritical (n<1)
        return 1e12
    ll = 0.0
    for ts in groups:
        R = 0.0
        prev = ts[0]
        s = math.log(mu)                       # first event: lambda = mu
        for k in range(1, len(ts)):
            R = math.exp(-beta * (ts[k] - prev)) * (1.0 + R)
            lam = mu + alpha * R
            if lam <= 0:
                return 1e12
            s += math.log(lam)
            prev = ts[k]
        integral = mu * T + sum((alpha / beta) * (1.0 - math.exp(-beta * (T - t))) for t in ts)
        ll += s - integral
    return -ll


def _poisson_ll(groups, T, n_events):
    """Best homogeneous-Poisson baseline log-likelihood (no self-excitation)."""
    G = len(groups)
    mu_p = n_events / (G * T)
    return n_events * math.log(mu_p) - G * mu_p * T, mu_p


def fit(path):
    groups, T, n_events = _load_groups(path)
    multi = [g for g in groups if len(g) >= 2]
    # initial guess: baseline ~ avg rate; moderate, fast-decaying excitation
    mu0 = max(n_events / (len(groups) * T), 1e-6)
    x0 = [mu0, 0.3, 1.0]
    bounds = [(1e-9, None), (1e-9, None), (1e-6, None)]
    res = minimize(_neg_ll, x0, args=(groups, T), method="L-BFGS-B", bounds=bounds)
    mu, alpha, beta = res.x
    ll_h = -res.fun
    ll_p, mu_p = _poisson_ll(groups, T, n_events)

    n_branch = alpha / beta                                  # branching factor
    half_life = math.log(2) / beta                          # minutes
    aic_h, aic_p = 2 * 3 - 2 * ll_h, 2 * 1 - 2 * ll_p
    params = {"mu": mu, "alpha": alpha, "beta": beta,
              "branching_factor": n_branch, "half_life_min": half_life,
              "loglik_hawkes": ll_h, "loglik_poisson": ll_p,
              "aic_hawkes": aic_h, "aic_poisson": aic_p,
              "self_exciting": aic_h < aic_p, "n_events": n_events,
              "n_locations": len(groups), "T_min": T}
    with open(f"{MODELS_DIR}/hawkes_params.pkl", "wb") as f:
        pickle.dump(params, f)
    return params


class HawkesModel:
    """Live cascade-risk intensity from fitted parameters."""
    def __init__(self, models_dir=MODELS_DIR):
        with open(f"{models_dir}/hawkes_params.pkl", "rb") as f:
            self.p = pickle.load(f)

    @property
    def branching_factor(self):
        return self.p["branching_factor"]

    @property
    def half_life_min(self):
        return self.p["half_life_min"]

    def intensity(self, minutes_since_recent_events):
        """Current intensity given minutes-ago of recent incidents at a location."""
        mu, alpha, beta = self.p["mu"], self.p["alpha"], self.p["beta"]
        exc = sum(alpha * math.exp(-beta * max(m, 0)) for m in minutes_since_recent_events)
        return mu + exc

    def risk_multiplier(self, minutes_since_recent_events):
        mu = max(self.p["mu"], 1e-4)        # floor to avoid explosion
        return self.intensity(minutes_since_recent_events) / mu

    def decay_curve(self, n_recent=1, horizon_min=180, step=5):
        """Risk multiplier over time after n simultaneous incidents (for plotting)."""
        mu, alpha, beta = self.p["mu"], self.p["alpha"], self.p["beta"]
        xs = list(range(0, horizon_min + 1, step))
        ys = [(mu + n_recent * alpha * math.exp(-beta * x)) / mu for x in xs]
        return xs, ys

    def expected_followons(self, horizon_min=60, n_recent=1):
        """ANTICIPATORY forecast: expected direct follow-on incidents triggered within the
        next `horizon_min` minutes after `n_recent` incidents at a location."""
        alpha, beta = self.p["alpha"], self.p["beta"]
        return n_recent * (alpha / beta) * (1 - math.exp(-beta * horizon_min))

    def expected_cluster_size(self):
        """Total expected incidents in a cluster seeded by one incident (cascades of
        cascades included): 1 / (1 - branching_factor)."""
        n = self.p["branching_factor"]
        return 1.0 / (1.0 - n) if n < 1 else float("inf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    p = fit(ap.parse_args().data)
    print(f"\n===== HAWKES self-exciting cascade model =====")
    print(f"  fitted on {p['n_events']} incidents across {p['n_locations']} locations")
    print(f"  baseline mu      = {p['mu']:.5f} incidents/min/location")
    print(f"  excitation alpha = {p['alpha']:.4f}")
    print(f"  decay beta       = {p['beta']:.4f} /min")
    print(f"\n  >> BRANCHING FACTOR n = {p['branching_factor']:.3f}")
    print(f"     (each incident triggers ~{p['branching_factor']:.2f} follow-on incidents on average)")
    print(f"  >> EXCITATION HALF-LIFE = {p['half_life_min']:.0f} min")
    print(f"     (elevated cascade risk roughly halves every {p['half_life_min']:.0f} minutes)")
    print(f"\n  Is it genuinely self-exciting (vs random Poisson)?")
    print(f"     Hawkes AIC {p['aic_hawkes']:.0f}  vs  Poisson AIC {p['aic_poisson']:.0f}  ->  "
          f"{'YES, self-exciting' if p['self_exciting'] else 'no'}")
    # anticipatory forecast (consequence of the fitted process)
    n = p["branching_factor"]; alpha, beta = p["alpha"], p["beta"]
    f60 = (alpha / beta) * (1 - math.exp(-beta * 60))
    cluster = 1.0 / (1.0 - n) if n < 1 else float("inf")
    print(f"\n  ANTICIPATORY FORECAST (from the same model):")
    print(f"     after an incident, expect ~{f60:.2f} follow-on incidents within 60 min")
    print(f"     expected total cluster size per incident: {cluster:.2f}")
    print(f"\n[saved] models/hawkes_params.pkl\n")