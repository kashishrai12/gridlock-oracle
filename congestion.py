"""
congestion.py — Congestion surface for capacity-aware diversion routing.

Two signals, combined into a 0..1 "load" for any point on the map:
  1. STRUCTURAL load  — where incidents historically cluster (grid density from the
                        full event log). Always available.
  2. LIVE load        — events ACTIVE at the query time near a point (optional, used
                        when a timestamp is supplied) so diversions also avoid roads
                        next to other ongoing incidents.

The router multiplies edge travel cost by (1 + alpha * load), so congested roads look
"longer" and get routed around. This is the project's differentiator: diversions that
don't create a second jam.

Run: python congestion.py --data data/flipkart_gridlock.csv
Output: models/congestion_grid.csv
"""

import argparse, os
import numpy as np
import pandas as pd

MODELS_DIR = "models"
CELL_DEG = 0.004          # ~440 m grid cell
GRID_PATH = f"{MODELS_DIR}/congestion_grid.csv"


def build_congestion_grid(data_path, cell_deg=CELL_DEG):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(data_path)
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("Event log needs latitude/longitude to build a congestion surface.")
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    # weight road-closure events more — they choke capacity harder
    if "requires_road_closure" in df.columns:
        w = 1 + 1.5 * (df["requires_road_closure"].astype(str).str.lower()
                       .isin(["1", "true", "yes", "y", "t"]).astype(int))
    else:
        w = pd.Series(1.0, index=df.index)
    df["w"] = w

    df["lat_bin"] = (df["latitude"] / cell_deg).round() * cell_deg
    df["lon_bin"] = (df["longitude"] / cell_deg).round() * cell_deg
    g = (df.groupby(["lat_bin", "lon_bin"])["w"].sum()
           .rename("load_raw").reset_index())
    # robust normalisation: 95th percentile -> 1.0, clipped
    cap = g["load_raw"].quantile(0.95) or 1.0
    g["load_norm"] = (g["load_raw"] / cap).clip(0, 1).round(4)
    g.to_csv(GRID_PATH, index=False)
    print(f"[congestion] {len(df)} geocoded events -> {len(g)} grid cells "
          f"(cell≈{cell_deg*111:.1f}km)")
    hot = g.sort_values("load_norm", ascending=False).head(1).iloc[0]
    print(f"[congestion] hottest cell load=1.0 at ({hot.lat_bin:.3f},{hot.lon_bin:.3f})")
    print(f"[congestion] saved {GRID_PATH}")
    return g


class CongestionSurface:
    """Query load at any point or along a route. Optionally factor in live events."""
    def __init__(self, grid_df, events_df=None, cell_deg=CELL_DEG):
        self.cell = cell_deg
        self.grid = {(round(r.lat_bin, 6), round(r.lon_bin, 6)): float(r.load_norm)
                     for r in grid_df.itertuples()}
        self.events = events_df          # optional df with latitude/longitude/start/end

    # -- structural --
    def _bin(self, x):
        return round(round(x / self.cell) * self.cell, 6)

    def load_at(self, lat, lon):
        return self.grid.get((self._bin(lat), self._bin(lon)), 0.0)

    def route_load(self, coords):
        if not coords:
            return 0.0
        return float(np.mean([self.load_at(la, lo) for la, lo in coords]))

    # -- live (optional) --
    def active_load(self, lat, lon, when, radius_m=700):
        """Normalised count of events active at `when` within radius_m of (lat,lon)."""
        if self.events is None or when is None:
            return 0.0
        ev = self.events
        when = pd.Timestamp(when)
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        active = ev[(ev["start"] <= when) & (ev["end"] >= when)]
        if active.empty:
            return 0.0
        d = _haversine_vec(lat, lon, active["latitude"].values, active["longitude"].values)
        n = int((d <= radius_m).sum())
        return float(min(n / 5.0, 1.0))      # 5+ concurrent nearby = saturated

    def combined_load(self, lat, lon, when=None):
        s = self.load_at(lat, lon)
        a = self.active_load(lat, lon, when)
        return float(min(0.6 * s + 0.7 * a + 0.4 * s * a, 1.0))


def _haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371000
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_surface(models_dir=MODELS_DIR, data_path=None):
    """Load the saved grid (and optionally the event log for live awareness)."""
    gp = f"{models_dir}/congestion_grid.csv"
    if not os.path.exists(gp):
        return None
    grid = pd.read_csv(gp)
    events = None
    if data_path and os.path.exists(data_path):
        try:
            ev = pd.read_csv(data_path)
            ev = ev[ev["latitude"].notna() & ev["longitude"].notna()].copy()
            ev["start"] = pd.to_datetime(ev["start_datetime"], errors="coerce", utc=True)
            end_col = "closed_datetime" if "closed_datetime" in ev else "end_datetime"
            ev["end"] = pd.to_datetime(ev.get(end_col), errors="coerce", utc=True)
            ev["end"] = ev["end"].fillna(ev["start"] + pd.Timedelta(hours=1))
            events = ev[["latitude", "longitude", "start", "end"]].dropna(subset=["start"])
        except Exception:
            events = None
    return CongestionSurface(grid, events)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    g = build_congestion_grid(ap.parse_args().data)
    surf = load_surface()
    if surf:
        # demo: load at the hottest cell vs a random edge
        hot = g.sort_values("load_norm", ascending=False).iloc[0]
        print(f"[demo] load at hottest cell: {surf.load_at(hot.lat_bin, hot.lon_bin):.2f}")