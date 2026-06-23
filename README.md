# Gridlock Oracle

**Predicting and pre-empting traffic-incident cascades in Bengaluru.**
Flipkart GRIDLOCK · Theme 2 : Event-Driven Congestion

Gridlock Oracle is a decision-support system for traffic control rooms. It doesn't just detect
incidents, it predicts the ones that will *compound*, targets scarce resources at them, routes
diversions on live traffic, and learns from every outcome. Built entirely on incident data the
city already collects, plus two live feeds. No new sensors.

🔗 **Live demo:** https://gridlock-oracle.streamlit.app/

---

## The core idea

Cities react to incidents one at a time and pre-position resources blindly. Two things are true
in the data: most deployments are wasted on incidents that never needed them, and **incidents
trigger more incidents nearby**. We built a system around both — modelling incidents as a
*self-exciting point process* (a Hawkes process, the same maths used for earthquake aftershocks)
to predict the cascade, and a calibrated, uncertainty-aware pipeline to target the response.

---

## Key results (real Bengaluru data, held-out where applicable)

| Capability | Result | Source |
|---|---|---|
| **Cascade (Hawkes) model** | branching factor **0.30**, half-life **27 min**; self-exciting confirmed (AIC 163,203 vs 180,325) | `hawkes.py` |
| **Closure-need classifier** | ROC-AUC 0.72, PR-AUC 0.348 (3.3x base rate) | `train_model.py` |
| **Operating backtest** | **49%** of closures caught at **17.8%** flagged -> **85.9% wasted deployments avoided**, **~6,800 officer-hours saved**, **2.8x** efficiency | `evaluate.py` |
| **Calibration** | ECE cut **~44%** (isotonic) | `calibrate.py` |
| **Conformal prediction** | 89.4% empirical coverage; commits on 93%, abstains on 7% | `conformal.py` |
| **Survival analysis** | median clearance 71 min; uses 3,459 events (vs regression's 2,724) | `survival.py` |
| **Optimizer scalability** | **347 ms** at 2,000 incidents (~28x a busy day), near-linear | `scale_benchmark.py` |
| **Learning loop** | ~67% calibration-error reduction with feedback | `learning_loop.py` |

Numbers are reported as **measured / held-out / estimated** .

---

## Leakage-free by design

Our first instinct — predicting a derived congestion score — leaked. We caught it early and
pivoted to predicting **closure-need**, a target known at incident-creation time. We then audited
every feature and split:

- **No post-event fields.** Any aggregate computed from closure outcomes, resolution time, or end
  status was removed. The model trains strictly on what is known the moment an incident is reported.
- **Encoders and calibrators fit on the training split only** — no information from the test set
  leaks backward into preprocessing.
- **Temporal held-out test** — we train on the past and test on the future, not a random shuffle.
  This is the honest split for a forecasting system, and a *stricter* test than a random split:
  it measures whether the model generalises to days it has never seen.

Every headline number above comes from this leakage-free, temporally held-out evaluation — which
is why our metrics are deliberately realistic rather than suspiciously perfect.

---

## Severity + confidence, not a black box

For any incident the system outputs a **severity tier** (Critical / High / Moderate / Low) together
with a **calibrated confidence** and a graded readiness recommendation (pre-position / standby /
monitor). Unlike a severity derived from a couple of raw columns, ours is grounded in a calibrated
closure-risk model with **conformal uncertainty bounds**. when the model is unsure it says so and
routes the incident to a human, with a 90% coverage guarantee.

Every prediction is **explainable**: the dashboard shows exactly which factors raised or lowered the
closure risk for that specific incident, computed with exact XGBoost SHAP values (no external
black-box dependency). One click exports a printable **commander briefing PDF** with the severity,
the recommended response, the reasoning, and historical context.

---

## Impact: same coverage, at a fraction of the cost

Framed as a counterfactual against the status quo:

- **Without targeting (blanket deployment):** resources are pre-positioned at every incident, and
  the vast majority of those deployments are wasted on incidents that never needed them.
- **With Gridlock Oracle:** the same closure coverage is achieved while flagging under 18% of
  incidents — **85.9% fewer wasted deployments** and an estimated **~6,800 officer-hours saved
  (69%)**, at **2.8x** the efficiency of blanket coverage.

Officer-hours are an estimate against a deliberately conservative blanket-coverage baseline. We
present the full operating curve instead of a single cherry picked point. The city chooses where to sit
on the recall-vs-efficiency trade-off.

---

## What's inside

**Live system**
- **TomTom incident feed**: real Bengaluru incidents, streamed and triaged live (`live_feed.py`)
- **Mappls live traffic**: congestion-aware diversion routing (`mappls.py`, `routing.py`)

**Intelligence**
- Calibrated **closure-need classifier** with **conformal** uncertainty (`predictor.py`, `conformal.py`)
- **Hawkes cascade** model + anticipatory forecast (`hawkes.py`)
- **Survival analysis** for clearance time (`survival.py`)
- **ILP optimizer** (scipy HiGHS) for officer/barricade allocation (`optimizer.py`)
- **Learning loop** that recalibrates from outcomes (`learning_loop.py`)
- Per-prediction **SHAP explanations** + one-click **commander briefing PDF** (`predictor.py`, `briefing.py`)

**Honest methodology**: we tested and *rejected* six approaches (clearance-time regression,
keyword NLP, character n-grams, multilingual embeddings, a spatiotemporal forecast, and a
circular cascade heuristic), keeping only what survived validation.

---

## Quick start

### Prerequisites
- Python 3.11 (recommended. Matches the pinned dependencies and saved model artifacts)
- conda or venv

### 1. Setup
```bash
conda create -n gridlock python=3.11 -y
conda activate gridlock
pip install -r requirements.txt
```

### 2. (Optional) live feeds — the app runs fully without these
```bash
# TomTom live incident feed (free key: https://developer.tomtom.com)
setx TOMTOM_API_KEY "your_key"          # Windows (open a NEW terminal after)
export TOMTOM_API_KEY="your_key"        # macOS/Linux

# Mappls live traffic routing
setx MAPPLS_CLIENT_ID "..."  ;  setx MAPPLS_CLIENT_SECRET "..."  ;  setx MAPPLS_REST_KEY "..."
```
Without keys, the Live Feed page falls back to **replay mode** and routing uses historical congestion.

### 3. Build the models (run once, in order)
```bash
python train_model.py    --data data/flipkart_gridlock.csv
python calibrate.py      --data data/flipkart_gridlock.csv
python analogs.py        --data models/enriched_dataset.csv
python hotspots.py       --data data/flipkart_gridlock.csv
python congestion.py     --data data/flipkart_gridlock.csv
python hawkes.py         --data data/flipkart_gridlock.csv
python survival.py       --data data/flipkart_gridlock.csv
python conformal.py      --data data/flipkart_gridlock.csv
```

### 4. (Optional) reproduce the headline numbers
```bash
python evaluate.py        --data data/flipkart_gridlock.csv   # backtest
python scale_benchmark.py                                     # ILP solve-time vs load
```

### 5. Launch the dashboard
```bash
streamlit run dashboard.py --server.port 8051
# open http://localhost:8501
```

---

## Dashboard pages

| Page | What it shows |
|---|---|
| **Predict Event** | Per-incident closure risk, impact, readiness tier, conformal confidence badge, SHAP explanation, briefing PDF |
| **Analytics** | Hotspots, clearance survival curves, distributions |
| **Event Cascades** | Hawkes branching factor + live decay curve |
| **Diversion & Barricades** | Capacity-aware reroutes on live Mappls traffic + barricade entry points |
| **Deployment Optimizer** | ILP allocation of limited officers/barricades + scalability benchmark |
| **Learning Loop** | Animated "watch it learn" + human-in-the-loop feedback |
| **Live Feed** | Real TomTom incidents (or replay) + live cascade-risk heatmap |

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SciPy (HiGHS) · Streamlit · Plotly · ReportLab ·
TomTom Traffic API · Mappls (MapmyIndia) API

---

## License & data

Built for Flipkart GRIDLOCK. The competition dataset is used under the event's terms and is not
redistributed beyond what the rules permit.
