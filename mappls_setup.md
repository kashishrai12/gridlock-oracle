# Mappls (MapmyIndia) Integration — Setup & Run Guide

This adds **live traffic** and **India-accurate routing/maps** to Gridlock Oracle. It is
designed to *enrich* the existing system: if the API key is missing or any call fails,
everything silently falls back to your current historical/OSM behaviour — the demo can
never break.

---

## What got integrated

1. **Live traffic on diversions** — each diversion route's congestion score is upgraded
   from *historical density* to a **live traffic factor** from Mappls, and routes are
   re-ranked by current conditions. A green "LIVE traffic-aware (Mappls)" badge appears.
2. **India-accurate maps** — the diversion map uses Mappls tiles with a **live-traffic
   overlay layer** (toggle in the layer control) instead of generic dark tiles.
3. **Production routing primitives** — `mappls.py` exposes real Indian-road routing and
   geocoding for any further use.

Files: `mappls.py` (new), `routing.py` (updated), `routing_page.py` (updated),
`requirements.txt` (new).

---

## Step 1 — Get free Mappls credentials

1. Sign in at **https://apis.mappls.com/console/**
2. Create a project. From it you need three values:
   - **client_id** and **client_secret** (OAuth)
   - a **REST API key** (used in route/tile URLs)
3. Enable these APIs on the project (free tier is enough for a demo): *Routing / Route Adv*,
   *Map Tiles*, *Traffic*, *Geocoding*.

## Step 2 — Set environment variables (PowerShell)

```powershell
setx MAPPLS_CLIENT_ID "your_client_id"
setx MAPPLS_CLIENT_SECRET "your_client_secret"
setx MAPPLS_REST_KEY "your_rest_api_key"
```
Close and reopen the terminal so they load. (Verify with `echo $Env:MAPPLS_REST_KEY`.)

## Step 3 — Install the one new dependency

```powershell
pip install requests
```

## Step 4 — Verify the integration (important)

```powershell
python mappls.py --selftest
```
You'll get a PASS/FAIL for token, routing, live-traffic, and geocoding. If a step FAILS,
the endpoint URL for *that* feature may differ in your API version — open `mappls.py`,
find the URL constant at the top (e.g. `ROUTE_URL`, `TRAFFIC_TILE_URL`), and update it
from your console docs. Each feature is independent, so fixing one won't affect the others.

## Step 5 — Run the dashboard

```powershell
streamlit run dashboard.py
```
Go to **Diversion & Barricades**, generate routes, and you should see:
- the Mappls map with a **Live Traffic** layer toggle,
- a green **LIVE traffic-aware** badge,
- each route card showing a `🟢 live N×` congestion factor.

If credentials aren't set, the page works exactly as before (dark tiles, historical
congestion) — no errors.

---

## How it works (for the pitch / Q&A)

- **No leakage of demo risk:** `full_routing_analysis()` computes diversions with your
  existing avoidance logic, then `_enrich_with_live_traffic()` overlays Mappls live data.
  The `try/except` means a network failure mid-demo just reverts to historical silently.
- **Live factor:** for each route we ask Mappls for the current driving time and compare it
  to a free-flow baseline; the ratio (≥1.0) is the live congestion multiplier used to rank.
- **Honest framing:** the *quantitative* live signal is per-route traffic time; the traffic
  *tiles* are a visual overlay. Say it that way and it's bulletproof.

## For your architecture slide (this is the feasibility win)

Even if you only demo the live routing, put Mappls on the architecture diagram as the
**"Live Traffic & Routing layer (Mappls APIs)"** feeding the diversion engine. That single
box answers the judge's "how does this deploy with live data?" question and lifts your
feasibility score — the integration makes it real, not aspirational.

## Cost / limits

Free tier covers demo-level usage. Calls are made only when a user generates routes (a few
per click), and OAuth tokens are cached, so you won't burn quota idling.