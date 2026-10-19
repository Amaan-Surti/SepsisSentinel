"""Online inference path for SepsisSentinel — shared by the demo cohort and the API.

raw hourly rows -> cleaning -> causal features -> LightGBM -> isotonic calibrator -> policy.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from sepsis.alerting import apply_policy
from sepsis.cleaning import clean
from sepsis.constants import RAW_COLUMNS
from sepsis.explain import explain_rows
from sepsis.features import build_features_from_df

logger = logging.getLogger(__name__)


@dataclass
class Artifacts:
    """Frozen final-model artifacts loaded from models/final/."""

    booster: lgb.Booster
    calibrator: Any
    policy: Dict[str, Any]
    feature_list: List[str]
    threshold_curve: List[Dict[str, float]]
    params: Dict[str, Any]
    explainer: Optional[shap.TreeExplainer] = None


def load_artifacts(
    model_dir: str | Path = "models/final", with_explainer: bool = True
) -> Artifacts:
    """Load booster, calibrator, policy, feature list and threshold curve."""
    model_dir = Path(model_dir)
    booster = lgb.Booster(model_file=str(model_dir / "lgbm.txt"))
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    with open(model_dir / "policy.json") as f:
        policy = json.load(f)
    with open(model_dir / "feature_list.json") as f:
        feature_list = json.load(f)
    with open(model_dir / "threshold_curve.json") as f:
        threshold_curve = json.load(f)
    with open(model_dir / "params.json") as f:
        params = json.load(f)
    explainer = shap.TreeExplainer(booster) if with_explainer else None
    logger.info("Loaded final artifacts from %s (%d features)", model_dir, len(feature_list))
    return Artifacts(booster, calibrator, policy, feature_list, threshold_curve, params, explainer)


def prepare_raw_rows(raw: pd.DataFrame, patient_id: str = "online") -> pd.DataFrame:
    """Fill missing raw columns with NaN, add carry columns and apply cleaning."""
    df = raw.copy()
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if "patient_id" not in df.columns:
        df["patient_id"] = patient_id
    if "hospital" not in df.columns:
        df["hospital"] = "NA"
    df["SepsisLabel"] = df["SepsisLabel"].fillna(0.0)
    df[RAW_COLUMNS] = df[RAW_COLUMNS].astype("float32")
    df["hour"] = df["hour"].astype("int16")
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    cleaned, _ = clean(df)
    return cleaned


def score_clean_rows(
    clean_df: pd.DataFrame, art: Artifacts, cfg: Dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features and return (features_df, scores) with calibrated risk per row."""
    feats = build_features_from_df(clean_df, cfg)
    raw_prob = art.booster.predict(feats[art.feature_list])
    risk = np.clip(art.calibrator.predict(raw_prob), 0.0, 1.0)
    scores = feats[["patient_id", "hour", "SepsisLabel"]].copy()
    scores["risk_raw"] = raw_prob
    scores["risk"] = risk
    return feats, scores


def alerts_for(scores: pd.DataFrame, policy: Dict[str, Any], threshold: float) -> np.ndarray:
    """Apply the k/sticky policy at a (possibly overridden) threshold."""
    return apply_policy(
        scores, "risk", threshold=threshold, k=int(policy["k"]), sticky=int(policy["sticky"])
    )


def reasons_for(feats: pd.DataFrame, art: Artifacts, top_n: int = 5) -> List[List[Dict]]:
    """Top-n positive SHAP reasons per row (requires a loaded explainer)."""
    if art.explainer is None:
        art.explainer = shap.TreeExplainer(art.booster)
    return explain_rows(art.explainer, feats[art.feature_list], top_n=top_n)
