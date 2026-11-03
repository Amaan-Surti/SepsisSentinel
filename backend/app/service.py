"""SepsisService: loads the served model and the demo cohort once, answers API queries.

MODEL_KIND=ensemble serves LightGBM + GRU (weight and policy from ensemble.json);
MODEL_KIND=lgbm serves the frozen LightGBM alone. SHAP reasons always come from LightGBM.
"""

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.app.settings import DISCLAIMER, Settings
from sepsis.config import load_config
from sepsis.inference import (
    alerts_for,
    load_artifacts,
    load_ensemble,
    prepare_raw_rows,
    reasons_for,
    score_clean_rows,
    score_clean_rows_ensemble,
)

logger = logging.getLogger(__name__)

MODEL_LABELS = {"ensemble": "LightGBM + GRU ensemble", "lgbm": "LightGBM"}
REASONS_SOURCE = {
    "ensemble": "LightGBM component of the ensemble (SHAP)",
    "lgbm": "LightGBM (SHAP)",
}
LOCKBOX_LABEL = "Lockbox test (LightGBM)"
DEV_CV_LABEL = "Development comparison (nested 5-fold CV)"

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


def _digest(path: Path) -> str:
    """First 8 hex digits of a file's SHA-256 (version tag)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


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
        self.kind = settings.model_kind
        self.cfg = load_config(settings.config_path)
        self.art = load_artifacts(settings.model_dir, with_explainer=True)
        self.lgbm_version = (
            f"lgbm-final-{self.art.params['num_boost_round']}r-"
            f"{_digest(settings.model_dir / 'lgbm.txt')}"
        )
        self.ens = None
        if self.kind == "ensemble":
            self.ens = load_ensemble(settings.ensemble_dir)
            self.policy = self.ens.policy
            self.model_version = (
                f"ensemble-w{self.ens.weight:.1f}-{self.lgbm_version}"
                f"+gru-{_digest(settings.ensemble_dir / 'gru.pt')}"
            )
            preds_file, suffix = "demo_predictions.parquet", "_ensemble"
        else:
            self.policy = self.art.policy
            self.model_version = self.lgbm_version
            preds_file, suffix = "demo_predictions_lgbm.parquet", ""

        demo = settings.demo_dir
        self.rows = pd.read_parquet(demo / "demo_rows.parquet").sort_values(["patient_id", "hour"])
        self.preds = pd.read_parquet(demo / preds_file)
        self.preds = self.preds.sort_values(["patient_id", "hour"]).reset_index(drop=True)
        self.patients = pd.read_csv(demo / "demo_patients.csv", dtype={"patient_id": str})
        # Outcome labels follow the served model's own alerts (same rules for both)
        self.patients["outcome"] = self.patients[f"outcome{suffix}"]
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
    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "model_kind": self.kind, "model_version": self.model_version}

    def config(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "k": int(self.policy["k"]),
            "sticky": int(self.policy["sticky"]),
            "max_hour": self.max_hour,
            "n_patients": len(self.patients),
            "model_kind": self.kind,
            "model_label": MODEL_LABELS[self.kind],
            "model_version": self.model_version,
            "weight_lgbm": self.ens.weight if self.ens is not None else None,
            "reasons_source": REASONS_SOURCE[self.kind],
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
        if self.ens is not None:
            feats, scores = score_clean_rows_ensemble(clean_df, self.art, self.ens, self.cfg)
        else:
            feats, scores = score_clean_rows(clean_df, self.art, self.cfg)
        alerts = alerts_for(scores, self.policy, self.threshold)
        reasons = reasons_for(feats.iloc[[-1]], self.art, top_n=5)[0]
        return {
            "model_kind": self.kind,
            "hours": scores["hour"].astype(int).tolist(),
            "risk": [float(r) for r in scores["risk"]],
            "alert": [bool(a) for a in alerts],
            "reasons_last_hour": reasons,
            "reasons_source": REASONS_SOURCE[self.kind],
        }

    def metrics(self) -> Dict[str, Any]:
        """Result tables written by the pipeline (never recomputed here).

        lockbox_test: the frozen LightGBM, evaluated once on the 8,068 lockbox patients.
        dev_cv: LightGBM vs GRU vs ensemble on development data, nested 5-fold CV. The
        two sections are separate on purpose: their numbers are not comparable.
        """
        td = self.settings.tables_dir

        def table(name: str) -> List[Dict[str, Any]]:
            return _records(pd.read_csv(td / name))

        dev_cv = None
        dev_files = ("comparison.csv", "weights.csv", "bootstrap.csv", "final.csv",
                     "selection.json")
        if all((td / f"ensemble_dev_{f}").exists() for f in dev_files):
            with open(td / "ensemble_dev_selection.json") as f:
                selection = json.load(f)
            dev_cv = {
                "label": DEV_CV_LABEL,
                "note": (
                    "Development data only (32,268 patients): calibration, blend weight and "
                    "policy for each fold were fit on the other 4 folds. Not lockbox numbers."
                ),
                "comparison": table("ensemble_dev_comparison.csv"),
                "weights": table("ensemble_dev_weights.csv"),
                "bootstrap": table("ensemble_dev_bootstrap.csv"),
                "final": table("ensemble_dev_final.csv"),
                "selection": selection,
            }
        return {
            "active_model": self.kind,
            "lockbox_test": {
                "label": LOCKBOX_LABEL,
                "model": "LightGBM (models/final, frozen)",
                "note": (
                    "8,068 held-out patients, evaluated once (P5.T6). This result belongs to "
                    "the LightGBM model alone; the ensemble has not been measured on it."
                ),
                "metrics": table("final_test_metrics.csv"),
                "baselines": table("final_test_baselines.csv"),
                "subgroups": table("final_test_subgroups.csv"),
                "policy": self.art.policy,
            },
            "dev_cv": dev_cv,
            "cross_hospital": table("cross_hospital.csv"),
            "ablations": table("ablations.csv"),
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
            "model_kind": self.kind,
            "model_version": self.model_version,
            "markdown": md,
            "sections": sections,
        }
