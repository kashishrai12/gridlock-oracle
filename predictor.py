"""
predictor.py  —  rewired for the closure-need classifier

GridlockPredictor.predict(event_dict) returns:
    closure_prob        P(event requires a road closure)  [validated, AUC ~0.72]
    impact_score        transparent 0-10 score from closure_prob + context
    impact_tier         LOW / MODERATE / HIGH / CRITICAL
    expected_clearance  DESCRIPTIVE historical clearance at this junction (NOT predicted)
    resources           barricading / personnel / diversion recommendation
    explanations        top feature contributions (pred_contribs, categorical-safe)

Usage:
    from predictor import GridlockPredictor
    p = GridlockPredictor()
    p.predict({"start_datetime":"2024-06-01 18:00", "priority":"High",
               "event_type":"unplanned", "event_cause":"Accident",
               "veh_type":"Truck", "junction":"JN_5", "corridor":"C_2", "zone":"Z_1",
               "police_station":"PS_10"})
"""

import pickle
import numpy as np
import xgboost as xgb

from utils import preprocess as pp


class GridlockPredictor:
    def __init__(self, models_dir="models"):
        with open(f"{models_dir}/location_stats.pkl", "rb") as f:
            self.stats = pickle.load(f)
        self.clf = xgb.XGBClassifier(enable_categorical=True)
        self.clf.load_model(f"{models_dir}/closure_clf.json")
        self.feats, self.cats = pp.classifier_feature_cols(self.stats)
        self.g_clear = self.stats["global"]["clear"]
        self.barricade_threshold = 0.30      # operating point chosen from PR curve

    # ------------------------------------------------------------------ #
    def predict(self, event: dict):
        X, enr = pp.featurize_event(event, self.stats)
        row = enr.iloc[0]
        p = float(self.clf.predict_proba(X)[:, 1][0])

        impact = pp.closure_impact_score(
            p, row.get("priority_encoded", 0), row.get("is_planned", 0),
            row.get("junction_hist_clearance", self.g_clear), self.g_clear)
        tier = pp.impact_tier(impact)

        return {
            "closure_prob": round(p, 3),
            "impact_score": impact,
            "impact_tier": tier,
            "expected_clearance_mins": int(round(row.get("junction_hist_clearance", self.g_clear))),
            "is_known_location": bool(row.get("is_known_junction", 0)),
            "resources": self._resources(p, impact, row),
            "explanations": self._explain(X),
        }

    # ------------------------------------------------------------------ #
    def _resources(self, p_closure, impact, row):
        """Recommendations grounded in the validated closure probability + impact.
        Every number traces to a stated rule, not a black box."""
        barricades_needed = p_closure >= self.barricade_threshold
        n_barricades = (4 if p_closure >= 0.6 else 2) if barricades_needed else 0
        personnel = int(np.clip(round(2 + impact * 1.2), 2, 16))
        supervisors = 1 + (impact >= 7)
        return {
            "barricading_recommended": barricades_needed,
            "barricades": n_barricades,
            "personnel": personnel,
            "supervisors": int(supervisors),
            "diversion_recommended": bool(p_closure >= 0.5 or impact >= 7),
            "rapid_response_required": bool(impact >= 8),
            "expected_clearance_mins": int(round(row.get("junction_hist_clearance", self.g_clear))),
        }

    # ------------------------------------------------------------------ #
    def _explain(self, X, top=5):
        """Per-event feature contributions via XGBoost pred_contribs (SHAP values).
        Works with categorical splits; no external shap dependency."""
        booster = self.clf.get_booster()
        dm = xgb.DMatrix(X, enable_categorical=True)
        contribs = booster.predict(dm, pred_contribs=True)[0]   # last entry = bias
        pairs = list(zip(self.feats, contribs[:-1]))
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return [{"feature": f, "contribution": round(float(c), 3),
                 "direction": "raises" if c > 0 else "lowers"} for f, c in pairs[:top]]


if __name__ == "__main__":
    p = GridlockPredictor()
    demo = {"start_datetime": "2024-06-01 18:00", "priority": "High",
            "event_type": "unplanned", "event_cause": "Accident", "veh_type": "Truck",
            "junction": "JN_5", "corridor": "C_2", "zone": "Z_1", "police_station": "PS_10"}
    import json
    print(json.dumps(p.predict(demo), indent=2))