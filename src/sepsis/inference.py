"""Online inference path for SepsisSentinel — shared by the demo cohort and the API.

LightGBM: raw hourly rows -> cleaning -> causal features -> LightGBM -> isotonic -> policy.
Ensemble: the same features feed LightGBM -> isotonic and, as sequence rows, a
unidirectional GRU (CPU, float32) -> its own isotonic; the two calibrated risks are
blended with the dev-chosen weight and the ensemble policy is applied. SHAP reasons
always come from the LightGBM component.
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
from sepsis.sequences import apply_extra_norm, apply_norm, rows_to_matrix

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


@dataclass
class EnsembleArtifacts:
    """Dev-only ensemble artifacts from models/ensemble_final/ (LightGBM stays in models/final)."""

    gru: Any
    norm: Dict[str, Any]
    extra_features: List[str]
    gru_calibrator: Any
    weight: float
    policy: Dict[str, Any]
    meta: Dict[str, Any]


def load_ensemble(ens_dir: str | Path = "models/ensemble_final") -> EnsembleArtifacts:
    """Load the final GRU (on CPU, eval mode), its calibrator and ensemble.json."""
    import torch

    from sepsis.train_gru import SepsisGRU

    ens_dir = Path(ens_dir)
    bundle = torch.load(ens_dir / "gru.pt", map_location="cpu", weights_only=True)
    gru = SepsisGRU(int(bundle["n_inputs"]), bundle["config"])
    gru.load_state_dict(bundle["state"])
    gru.eval()
    with open(ens_dir / "ensemble.json") as f:
        meta = json.load(f)
    logger.info("Loaded ensemble from %s (w_lgbm=%.2f)", ens_dir, meta["weight_lgbm"])
    return EnsembleArtifacts(
        gru=gru,
        norm=bundle["norm"],
        extra_features=list(bundle.get("extra_features", [])),
        gru_calibrator=joblib.load(ens_dir / "gru_calibrator.joblib"),
        weight=float(meta["weight_lgbm"]),
        policy=meta["policy"],
        meta=meta,
    )


def gru_inputs(feats: pd.DataFrame, ens: EnsembleArtifacts) -> np.ndarray:
    """Normalised GRU input matrix for feature rows (sorted by patient, hour)."""
    X = apply_norm(rows_to_matrix(feats), ens.norm)
    if ens.extra_features:
        E = apply_extra_norm(feats[ens.extra_features].to_numpy("float32"), ens.norm["extra"])
        X = np.hstack([X, E])
    return X


def gru_raw_probs(feats: pd.DataFrame, ens: EnsembleArtifacts) -> np.ndarray:
    """Uncalibrated GRU probability per row, one stay at a time on the CPU."""
    import torch

    pids = feats["patient_id"].to_numpy()
    hours = feats["hour"].to_numpy()
    edges = np.r_[0, np.flatnonzero(pids[1:] != pids[:-1]) + 1, len(feats)]
    X = gru_inputs(feats, ens)
    out = np.empty(len(feats), dtype="float64")
    with torch.no_grad():
        for s, e in zip(edges[:-1], edges[1:]):
            if (np.diff(hours[s:e]) <= 0).any():
                raise ValueError("Feature rows must be sorted by hour within each patient")
            logits = ens.gru(torch.from_numpy(X[s:e])[None])[0]
            out[s:e] = torch.sigmoid(logits).numpy()
    return out


def score_clean_rows_ensemble(
    clean_df: pd.DataFrame, art: Artifacts, ens: EnsembleArtifacts, cfg: Dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features plus per-row LightGBM, GRU and blended calibrated risk ("risk")."""
    feats, scores = score_clean_rows(clean_df, art, cfg)
    scores = scores.rename(columns={"risk_raw": "lgbm_raw", "risk": "risk_lgbm"})
    scores["gru_raw"] = gru_raw_probs(feats, ens)
    scores["risk_gru"] = np.clip(ens.gru_calibrator.predict(scores["gru_raw"].to_numpy()), 0, 1)
    w = ens.weight
    scores["risk"] = w * scores["risk_lgbm"] + (1 - w) * scores["risk_gru"]
    return feats, scores
