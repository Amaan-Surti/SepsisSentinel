"""SepsisService: loads frozen artifacts and the demo cohort once, answers API queries."""

import hashlib
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.app.settings import DISCLAIMER, Settings
from sepsis.config import load_config
from sepsis.inference import (
    alerts_for,
    load_artifacts,
    prepare_raw_rows,
    reasons_for,
    score_clean_rows,
)

logger = logging.getLogger(__name__)

VITALS_SHOWN = ["HR", "MAP", "SBP", "Resp", "Temp", "O2Sat"]
LABS_SHOWN = ["Lactate", "WBC", "Creatinine", "Platelets"]
WARD_VITALS = ["HR", "MAP", "Temp", "Resp", "WBC"]
TREND_HOURS = 12


class PatientNotFound(KeyError):
    """Raised for an unknown demo patient id."""


def _none(x: Any) -> Optional[float]:
    """Convert NaN/NA to None for JSON."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    return float(x)


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame -> JSON-safe list of dicts (NaN -> None)."""
    return [
        {k: (_none(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
        for r in df.to_dict(orient="records")
    ]


class SepsisService:
    """Holds the model, calibrator, policy and demo tables in memory."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = load_config(settings.config_path)
        self.art = load_artifacts(settings.model_dir, with_explainer=True)
        self.policy = self.art.policy
        digest = hashlib.sha256((settings.model_dir / "lgbm.txt").read_bytes()).hexdigest()[:8]
        self.model_version = f"lgbm-final-{self.art.params['num_boost_round']}r-{digest}"

        demo = settings.demo_dir
        self.rows = pd.read_parquet(demo / "demo_rows.parquet").sort_values(["patient_id", "hour"])
        self.preds = pd.read_parquet(demo / "demo_predictions.parquet")
        self.preds = self.preds.sort_values(["patient_id", "hour"]).reset_index(drop=True)
        self.patients = pd.read_csv(demo / "demo_patients.csv", dtype={"patient_id": str})
        self.patients = self.patients.set_index("patient_id", drop=False)
        self.max_hour = int(self.patients["stay_hours"].max()) - 1

        self._rows_by_pid = {
            p: g.reset_index(drop=True) for p, g in self.rows.groupby("patient_id")
        }
        self._pred_idx = {p: g.index.to_numpy() for p, g in self.preds.groupby("patient_id")}
        self._alert_cache: Dict[float, np.ndarray] = {}
        logger.info("Service ready: %s, %d demo beds", self.model_version, len(self.patients))

    # ------------------------------------------------------------------ helpers
    @property
    def threshold(self) -> float:
        return float(self.policy["threshold"])

    def _alerts(self, threshold: float) -> np.ndarray:
        """Alerts for all demo rows at a threshold (same k/sticky as policy.json)."""
        key = round(float(threshold), 6)
        if key not in self._alert_cache:
            if len(self._alert_cache) > 256:
                self._alert_cache.clear()
            self._alert_cache[key] = alerts_for(self.preds, self.policy, key).astype(bool)
        return self._alert_cache[key]

    def _patient_state(self, pid: str, t: int, threshold: float) -> Dict[str, Any]:
        """Risk/alert/level for one patient at replay hour t."""
        p = self.patients.loc[pid]
        stay = int(p["stay_hours"])
        h = min(t, stay - 1)
        idx = self._pred_idx[pid]
        risk = self.preds["risk"].to_numpy()[idx]
        alert = self._alerts(threshold)[idx]
        since = None
        if alert[h]:
            since = h
            while since > 0 and alert[since - 1]:
                since -= 1
        level = "ALERT" if alert[h] else ("WATCH" if risk[h] >= 0.5 * threshold else "OK")
        return {
            "p": p,
            "h": h,
            "status": "active" if t < stay else "discharged",
            "risk": risk,
            "alert": alert,
            "alert_since": since,
            "level": level,
        }

    @staticmethod
    def _base_fields(p: pd.Series) -> Dict[str, Any]:
        return {
            "patient_id": str(p["patient_id"]),
            "display_name": str(p["display_name"]),
            "age": int(p["age"]),
            "gender": int(p["gender"]),
            "unit": int(p["unit"]),
        }

    # ------------------------------------------------------------------ public API
    def config(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "k": int(self.policy["k"]),
            "sticky": int(self.policy["sticky"]),
            "max_hour": self.max_hour,
            "n_patients": len(self.patients),
            "model_version": self.model_version,
            "disclaimer": DISCLAIMER,
        }

    def ward_view(self, t: int, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """All demo beds at hour t, active beds by risk desc, discharged beds last."""
        tau = self.threshold if threshold is None else threshold
        beds = []
        for pid in self.patients.index:
            s = self._patient_state(pid, t, tau)
            h = s["h"]
            rows = self._rows_by_pid[pid].iloc[: h + 1]
            vitals = {v: _none(rows[v].ffill().iloc[-1]) for v in WARD_VITALS}
            beds.append(
                {
                    **self._base_fields(s["p"]),
                    "hour": h,
                    "status": s["status"],
                    "risk": float(s["risk"][h]),
                    "risk_trend": [
                        float(r) for r in s["risk"][max(0, h - TREND_HOURS + 1) : h + 1]
                    ],
                    "alert": bool(s["alert"][h]),
                    "alert_since": s["alert_since"],
                    "level": s["level"],
                    "vitals": vitals,
                }
            )
        beds.sort(key=lambda b: (b["status"] == "discharged", -b["risk"]))
        return beds

    def patient_view(
        self, pid: str, t: int, threshold: Optional[float] = None, show_truth: bool = False
    ) -> Dict[str, Any]:
        """Trajectory of one patient up to hour t with top-5 reasons at that hour."""
        if pid not in self.patients.index:
            raise PatientNotFound(pid)
        tau = self.threshold if threshold is None else threshold
        s = self._patient_state(pid, t, tau)
        p, h = s["p"], s["h"]
        rows = self._rows_by_pid[pid].iloc[: h + 1]
        reasons = json.loads(self.preds.loc[self._pred_idx[pid][h], "top_reasons_json"])
        out = {
            **self._base_fields(p),
            "stay_hours": int(p["stay_hours"]),
            "hour": h,
            "status": s["status"],
            "level": s["level"],
            "alert_since": s["alert_since"],
            "threshold": tau,
            "hours": rows["hour"].astype(int).tolist(),
            "vitals": {v: [_none(x) for x in rows[v]] for v in VITALS_SHOWN},
            "labs": {v: [_none(x) for x in rows[v]] for v in LABS_SHOWN},
            "risk": [float(r) for r in s["risk"][: h + 1]],
            "alert": [bool(a) for a in s["alert"][: h + 1]],
            "reasons": reasons,
        }
        if show_truth:
            out["septic"] = bool(p["septic"])
            out["outcome"] = str(p["outcome"])
            out["onset_hour"] = int(p["t_sepsis"]) if p["septic"] else None
        return out

    def predict_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Online path: raw rows -> clean -> features -> model -> calibrator -> policy."""
        raw = pd.DataFrame(rows)
        if raw["hour"].duplicated().any():
            raise ValueError("Duplicate hour values in rows")
        clean_df = prepare_raw_rows(raw)
        feats, scores = score_clean_rows(clean_df, self.art, self.cfg)
        alerts = alerts_for(scores, self.policy, self.threshold)
        reasons = reasons_for(feats.iloc[[-1]], self.art, top_n=5)[0]
        return {
            "hours": scores["hour"].astype(int).tolist(),
            "risk": [float(r) for r in scores["risk"]],
            "alert": [bool(a) for a in alerts],
            "reasons_last_hour": reasons,
        }

    def metrics(self) -> Dict[str, Any]:
        """Frozen result tables (never recomputed)."""
        td = self.settings.tables_dir

        def table(name: str) -> List[Dict[str, Any]]:
            return _records(pd.read_csv(td / name))

        return {
            "test_metrics": table("final_test_metrics.csv"),
            "baselines": table("final_test_baselines.csv"),
            "cross_hospital": table("cross_hospital.csv"),
            "ablations": table("ablations.csv"),
            "subgroups": table("final_test_subgroups.csv"),
            "dev_oof": table("lgbm_dev_oof.csv"),
            "threshold_curve": self.art.threshold_curve,
            "policy": self.policy,
        }

    def model_card(self) -> Dict[str, Any]:
        """Model card markdown plus a heading -> body map."""
        md = self.settings.model_card.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", md, flags=re.M)
        sections: Dict[str, str] = {}
        for block in re.split(r"^## ", md, flags=re.M)[1:]:
            head, _, body = block.partition("\n")
            sections[head.strip()] = body.strip()
        return {
            "title": title_match.group(1).strip() if title_match else "Model card",
            "model_version": self.model_version,
            "markdown": md,
            "sections": sections,
        }
