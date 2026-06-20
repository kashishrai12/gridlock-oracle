

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

        # data-grounded resource estimates via historical analog retrieval (optional)
        self.analogs = None
        try:
            from analogs import AnalogRetriever
            self.analogs = AnalogRetriever(models_dir)
        except Exception:
            self.analogs = None

        # probability calibrator (optional) — makes closure_prob mean what it says
        self.calibrator = None
        try:
            with open(f"{models_dir}/closure_calibrator.pkl", "rb") as f:
                self.calibrator = pickle.load(f)
        except Exception:
            self.calibrator = None

    # ------------------------------------------------------------------ #
    def predict(self, event: dict):
        X, enr = pp.featurize_event(event, self.stats)
        row = enr.iloc[0]
        p_raw = float(self.clf.predict_proba(X)[:, 1][0])
        p = self._calibrate(p_raw)

        impact = pp.closure_impact_score(
            p, row.get("priority_encoded", 0), row.get("is_planned", 0),
            row.get("junction_hist_clearance", self.g_clear), self.g_clear)
        tier = pp.impact_tier(impact)

        # data-grounded path: retrieve similar historical incidents
        analog_block = None
        if self.analogs is not None:
            try:
                analog_block = self.analogs.query(event, closure_prob=p)
            except Exception:
                analog_block = None

        if analog_block is not None:
            expected_clearance = analog_block["expected_clearance_mins"]
            resources = analog_block["resources"]
        else:
            expected_clearance = int(round(row.get("junction_hist_clearance", self.g_clear)))
            resources = self._resources(p, impact, row)

        out = {
            "closure_prob": round(p, 3),
            "closure_prob_raw": round(p_raw, 3),
            "impact_score": impact,
            "impact_tier": tier,
            "readiness_tier": pp.readiness_tier(p),
            "expected_clearance_mins": int(expected_clearance),
            "is_known_location": bool(row.get("is_known_junction", 0)),
            "resources": resources,
            "explanations": self._explain(X),
        }
        if analog_block is not None:
            out["analogs"] = {
                "n_matched": analog_block["n_matched"],
                "clearance_p25": analog_block["clearance_p25"],
                "clearance_p75": analog_block["clearance_p75"],
                "analog_closure_rate": analog_block["analog_closure_rate"],
                "examples": analog_block["analogs"],
            }
        return out

    # ------------------------------------------------------------------ #
    def _calibrate(self, p):
        """Map the model's raw score to a calibrated probability (if a calibrator
        was fit). Falls back to the raw value when none is available."""
        c = self.calibrator
        if not c or c.get("model") is None or c.get("method") in (None, "raw"):
            return p
        try:
            if c["method"] == "isotonic":
                return float(np.clip(c["model"].predict([p])[0], 0.0, 1.0))
            if c["method"] == "platt":
                return float(c["model"].predict_proba([[p]])[0, 1])
        except Exception:
            return p
        return p

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