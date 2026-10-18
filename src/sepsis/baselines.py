"""Baseline models for SepsisSentinel — P2.T5.

B0: Never alert (sanity check)
B1: SIRS rule (score__sirs >= 2)
B2: Partial-qSOFA rule (score__qsofa_partial >= 2)
B3: Logistic regression on current values + measurement flags + clinical scores

B1/B2/B3 require features from Phase 3. This module is coded now but run after P3.T2.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sepsis.alerting import apply_policy, search_policy
from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED
from sepsis.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  B0: Never alert
# ---------------------------------------------------------------------------


def b0_never_alert(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """B0: All-zero predictions. Utility should be exactly 0.0."""
    if cfg is None:
        cfg = load_config()
    df = df.copy()
    df["prob_b0"] = 0.0
    df["alert_b0"] = 0
    return evaluate_predictions(df, "prob_b0", "alert_b0", cfg)


# ---------------------------------------------------------------------------
#  B1: SIRS rule
# ---------------------------------------------------------------------------


def b1_sirs_rule(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """B1: Alert when score__sirs >= 2."""
    if cfg is None:
        cfg = load_config()
    df = df.copy()
    df["prob_b1"] = (df["score__sirs"] >= 2).astype(float)
    df["alert_b1"] = (df["score__sirs"] >= 2).astype(int)
    return evaluate_predictions(df, "prob_b1", "alert_b1", cfg)


# ---------------------------------------------------------------------------
#  B2: Partial-qSOFA rule
# ---------------------------------------------------------------------------


def b2_qsofa_rule(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """B2: Alert when score__qsofa_partial >= 2."""
    if cfg is None:
        cfg = load_config()
    df = df.copy()
    df["prob_b2"] = (df["score__qsofa_partial"] >= 2).astype(float)
    df["alert_b2"] = (df["score__qsofa_partial"] >= 2).astype(int)
    return evaluate_predictions(df, "prob_b2", "alert_b2", cfg)


# ---------------------------------------------------------------------------
#  B3: Logistic regression (5-fold OOF)
# ---------------------------------------------------------------------------


def _get_b3_features(cfg: Optional[Dict[str, Any]] = None):
    """Return the list of feature columns for B3."""
    # 34 LOCF current values (original raw columns, LOCF-filled)
    raw_cols = [
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "DBP",
        "Resp",
        "EtCO2",
        "BaseExcess",
        "HCO3",
        "FiO2",
        "pH",
        "PaCO2",
        "SaO2",
        "AST",
        "BUN",
        "Alkalinephos",
        "Calcium",
        "Chloride",
        "Creatinine",
        "Bilirubin_direct",
        "Glucose",
        "Lactate",
        "Magnesium",
        "Phosphate",
        "Potassium",
        "Bilirubin_total",
        "TroponinI",
        "Hct",
        "Hgb",
        "PTT",
        "WBC",
        "Fibrinogen",
        "Platelets",
    ]
    # 34 measurement flags
    meas_cols = [f"{c}__meas" for c in raw_cols]
    # clinical scores and static features
    extra_cols = [
        "score__sirs",
        "score__sofa_partial",
        "idx__shock_index",
        "static__age",
        "static__gender",
    ]
    return raw_cols + meas_cols + extra_cols


def b3_logistic_regression(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """B3: Logistic regression with 5-fold OOF, alert policy by search_policy.

    Args:
        features_df: Feature matrix with patient_id, hour, SepsisLabel, and all feature cols.
        folds_df: Fold assignments with patient_id and fold columns.
        cfg: Config dict.

    Returns:
        (metrics_dict, oof_df) where oof_df has patient_id, hour, SepsisLabel, prob_b3, alert_b3.
    """
    if cfg is None:
        cfg = load_config()
    set_seeds(SEED)

    feat_cols = _get_b3_features(cfg)
    # Ensure all feature columns exist
    available = [c for c in feat_cols if c in features_df.columns]
    if len(available) < len(feat_cols):
        missing = set(feat_cols) - set(available)
        logger.warning("Missing %d feature columns for B3: %s", len(missing), missing)
    feat_cols = available

    # Merge fold info
    fold_map = folds_df.set_index("patient_id")["fold"].to_dict()
    features_df = features_df.copy()
    features_df["fold"] = features_df["patient_id"].map(fold_map)

    # Only dev patients
    features_df = features_df[features_df["fold"].notna()].copy()
    features_df["fold"] = features_df["fold"].astype(int)

    # OOF predictions
    oof_probs = np.full(len(features_df), np.nan)
    n_folds = folds_df["fold"].nunique()

    for fold_idx in range(n_folds):
        train_mask = features_df["fold"] != fold_idx
        val_mask = features_df["fold"] == fold_idx

        X_train = features_df.loc[train_mask, feat_cols].values
        y_train = features_df.loc[train_mask, "SepsisLabel"].values.astype(int)
        X_val = features_df.loc[val_mask, feat_cols].values

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        oof_probs[val_mask.values] = pipe.predict_proba(X_val)[:, 1]

        logger.info("B3 fold %d: train=%d, val=%d", fold_idx, train_mask.sum(), val_mask.sum())

    features_df["prob_b3"] = oof_probs

    # Search for best alert policy on OOF
    logger.info("Searching alert policy for B3...")
    best_params, grid_df = search_policy(features_df, "prob_b3", cfg)
    features_df["alert_b3"] = apply_policy(
        features_df,
        "prob_b3",
        best_params["threshold"],
        best_params["k"],
        best_params["sticky"],
    )

    metrics = evaluate_predictions(features_df, "prob_b3", "alert_b3", cfg)
    metrics["best_threshold"] = best_params["threshold"]
    metrics["best_k"] = best_params["k"]
    metrics["best_sticky"] = best_params["sticky"]

    oof_df = features_df[["patient_id", "hour", "SepsisLabel", "prob_b3", "alert_b3"]].copy()

    return metrics, oof_df


# ---------------------------------------------------------------------------
#  Run all baselines
# ---------------------------------------------------------------------------


def run_all_baselines(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run all baselines and save comparison table.

    Args:
        features_df: Full feature matrix for dev patients.
        folds_df: Fold assignments.
        cfg: Config dict.

    Returns:
        DataFrame with metrics per baseline.
    """
    if cfg is None:
        cfg = load_config()

    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Filter features to dev patients
    dev_pids = set(folds_df["patient_id"])
    features_df = features_df[features_df["patient_id"].isin(dev_pids)].copy()
    logger.info(
        "Evaluating baselines on dev cohort: %d patients, %d rows", len(dev_pids), len(features_df)
    )

    results = []

    # B0: Never alert
    logger.info("Running B0: Never alert...")
    m0 = b0_never_alert(features_df, cfg)
    m0["baseline"] = "B0_never_alert"
    results.append(m0)
    logger.info("B0 utility_norm = %.4f", m0["utility_norm"])

    # B1: SIRS
    if "score__sirs" in features_df.columns:
        logger.info("Running B1: SIRS rule...")
        m1 = b1_sirs_rule(features_df, cfg)
        m1["baseline"] = "B1_sirs"
        results.append(m1)
        logger.info("B1 utility_norm = %.4f", m1["utility_norm"])
    else:
        logger.warning("Skipping B1: score__sirs not found in features")

    # B2: qSOFA
    if "score__qsofa_partial" in features_df.columns:
        logger.info("Running B2: Partial-qSOFA rule...")
        m2 = b2_qsofa_rule(features_df, cfg)
        m2["baseline"] = "B2_qsofa"
        results.append(m2)
        logger.info("B2 utility_norm = %.4f", m2["utility_norm"])
    else:
        logger.warning("Skipping B2: score__qsofa_partial not found in features")

    # B3: Logistic regression
    logger.info("Running B3: Logistic regression...")
    m3, oof_df = b3_logistic_regression(features_df, folds_df, cfg)
    m3["baseline"] = "B3_logreg"
    results.append(m3)
    logger.info("B3 utility_norm = %.4f, AUROC = %.4f", m3["utility_norm"], m3["auroc"])

    report = pd.DataFrame(results)
    out_path = tables_dir / "baselines_dev_oof.csv"
    report.to_csv(out_path, index=False)
    logger.info("Baselines report saved to %s", out_path)

    return report


def main() -> None:
    """CLI entrypoint — run after Phase 3 features are built."""
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    cfg = load_config(args.config)
    set_seeds(SEED)

    features_path = Path(cfg["paths"]["features"])
    folds_path = Path(cfg["paths"]["folds"])

    if not features_path.exists():
        logger.error("Features file not found: %s — run Phase 3 first.", features_path)
        return
    if not folds_path.exists():
        logger.error("Folds file not found: %s — run splits first.", folds_path)
        return

    features_df = pd.read_parquet(features_path)
    folds_df = pd.read_csv(folds_path)

    run_all_baselines(features_df, folds_df, cfg)


if __name__ == "__main__":
    main()
