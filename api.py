"""
api.py — FastAPI backend
Usage: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predictor import GridlockPredictor
from routing import full_routing_analysis

app = FastAPI(title="Gridlock Oracle API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

predictor = GridlockPredictor(model_dir='models/')


class EventInput(BaseModel):
    event_type: str = "unplanned"          # "planned" | "unplanned"
    priority: str = "Medium"               # "High" | "Medium" | "Low"
    requires_road_closure: bool = False
    start_datetime: Optional[str] = None   # ISO format
    duration_mins: Optional[float] = 60
    junction: Optional[str] = None
    corridor: Optional[str] = None
    zone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None


class FeedbackInput(BaseModel):
    event_id: str
    predicted_score: float
    actual_personnel_deployed: int
    actual_clearance_mins: int
    officer_rating: int  # 1-5


# ---- Routes ----

@app.get("/")
def root():
    return {"status": "Gridlock Oracle is live", "version": "1.0"}


@app.post("/predict")
def predict_event(event: EventInput):
    try:
        result = predictor.predict(event.dict())
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/junctions/risk")
def junction_risk(top_n: int = 15):
    df = predictor.get_junction_leaderboard(top_n)
    return df.to_dict(orient='records')


@app.get("/corridors/load")
def corridor_load(top_n: int = 15):
    df = predictor.get_corridor_leaderboard(top_n)
    return df.to_dict(orient='records')


@app.get("/junctions/list")
def list_junctions():
    return predictor.junction_risk['junction'].dropna().tolist()


@app.get("/corridors/list")
def list_corridors():
    return predictor.corridor_stats['corridor'].dropna().tolist()


@app.get("/zones/list")
def list_zones():
    return predictor.zone_stats['zone'].dropna().tolist()


@app.post("/feedback")
def submit_feedback(fb: FeedbackInput):
    # Append to feedback log CSV
    row = fb.dict()
    log_path = 'models/feedback_log.csv'
    new_row = pd.DataFrame([row])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(log_path, index=False)
    return {"status": "feedback recorded", "total_records": len(updated)}


@app.get("/feedback/stats")
def feedback_stats():
    log_path = 'models/feedback_log.csv'
    if not os.path.exists(log_path):
        return {"total": 0, "avg_rating": None, "avg_clearance_error_mins": None}
    df = pd.read_csv(log_path)
    return {
        "total": len(df),
        "avg_rating": round(df['officer_rating'].mean(), 2),
        "avg_clearance_error_mins": round(
            (df['actual_clearance_mins'] - df['predicted_score'] * 8).abs().mean(), 1
        )
    }


# ---- Routing Endpoints ----

class RoutingInput(BaseModel):
    latitude: float
    longitude: float
    requires_road_closure: bool = True
    radius_m: int = 350


@app.post("/routing/full")
def routing_full(inp: RoutingInput):
    try:
        result = full_routing_analysis(
            lat=inp.latitude,
            lon=inp.longitude,
            requires_closure=inp.requires_road_closure,
            radius_m=inp.radius_m
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full")
def predict_and_route(event: EventInput):
    """Combined endpoint: prediction + routing in one call for the demo."""
    try:
        prediction = predictor.predict(event.dict())
        routing = None
        if event.latitude and event.longitude:
            routing = full_routing_analysis(
                lat=event.latitude,
                lon=event.longitude,
                requires_closure=event.requires_road_closure,
                radius_m=350
            )
        return {"status": "ok", "prediction": prediction, "routing": routing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))