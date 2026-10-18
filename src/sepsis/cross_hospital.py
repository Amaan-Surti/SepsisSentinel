"""Cross-hospital generalization study — Phase 4.

Evaluates how well models trained on Hospital A generalize to Hospital B, and vice versa.
Compares internal 5-fold CV reference against external zero-shot transfer
across 4 robustness variants:
  - R0: Full features
  - R1: Without missingness features (no __meas, __tsl, __cnt24)
  - R2: Per-hospital standardization (z-scored MEASURED variables within each hospital)
  - R3: Without administrative timing features (no static__hosp_adm_time, static__iculos)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from sepsis.alerting import apply_policy, search_policy
from sepsis.config import load_config, set_seeds
from sepsis.constants import MEASURED, SEED
from sepsis.features import build_features
from sepsis.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


def assign_hospital_folds(df: pd.DataFrame, hospital: str) -> pd.DataFrame:
    """Assign 5-fold stratified CV within all patients of a single hospital."""
    hosp_df = df[df["hospital"] == hospital].copy()
    pt = hosp_df.groupby("patient_id")["SepsisLabel"].max().reset_index()
    pt["septic"] = (pt["SepsisLabel"] == 1).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pt["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(pt, pt["septic"])):
        pt.loc[pt.index[val_idx], "fold"] = fold_idx

    fold_map = pt.set_index("patient_id")["fold"].to_dict()
    hosp_df["fold"] = hosp_df["patient_id"].map(fold_map).astype(int)
    return hosp_df


def get_standardized_features(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build or load per-hospital standardized features (R2)."""
    r2_path = Path("data/processed/features_r2.parquet")
    if r2_path.exists():
        logger.info("Loading existing standardized features from %s...", r2_path)
        return pd.read_parquet(r2_path)

    logger.info("Building per-hospital standardized features for R2...")
    clean_path = Path(cfg["paths"]["clean_rows"])
    clean_df = pd.read_parquet(clean_path)
    clean_df_std = clean_df.copy()

    # Preserve unstandardized copies for clinical scores/indices
    for var in MEASURED:
        clean_df_std[f"raw__{var}"] = clean_df_std[var]

    # Standardize each MEASURED variable within each hospital independently
    for hosp in ["A", "B"]:
        mask = clean_df_std["hospital"] == hosp
        for var in MEASURED:
            vals = clean_df_std.loc[mask, var]
            mu = vals.mean()
            sigma = vals.std()
            if sigma > 0:
                clean_df_std.loc[mask, var] = ((vals - mu) / sigma).astype("float32")

    cfg_r2 = json.loads(json.dumps(cfg))
    cfg_r2["paths"]["features"] = str(r2_path)
    features_r2 = build_features(clean_df_std, cfg_r2)
    logger.info("R2 standardized features saved to %s", r2_path)
    return features_r2


def run_hospital_internal_cv(
    hosp_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], IsotonicRegression, Dict[str, Any], float, pd.DataFrame]:
    """Train 5-fold CV within a single hospital and determine calibrated alert policy.

    Returns:
        (metrics_dict, calibrator, best_policy, mean_best_iter, oof_df)
    """
    cat_features = ["static__unit"] if "static__unit" in feature_cols else []
    oof_probs = np.full(len(hosp_df), np.nan)
    best_iters = []

    train_params = dict(params)
    train_params.pop("early_stopping_round", None)
    train_params.pop("early_stopping_rounds", None)
    train_params.pop("num_boost_round", None)
    train_params.update(
        {
            "seed": SEED,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
            "n_jobs": -1,
        }
    )

    for fold_idx in range(5):
        train_mask = hosp_df["fold"] != fold_idx
        val_mask = hosp_df["fold"] == fold_idx

        X_train = hosp_df.loc[train_mask, feature_cols]
        y_train = hosp_df.loc[train_mask, "SepsisLabel"].astype(int)
        X_val = hosp_df.loc[val_mask, feature_cols]
        y_val = hosp_df.loc[val_mask, "SepsisLabel"].astype(int)

        dtrain = lgb.Dataset(
            X_train, y_train, categorical_feature=cat_features, free_raw_data=False
        )
        dval = lgb.Dataset(
            X_val, y_val, categorical_feature=cat_features, free_raw_data=False, reference=dtrain
        )

        booster = lgb.train(
            train_params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
            ],
        )

        best_iter = booster.best_iteration
        best_iters.append(best_iter)
        preds = booster.predict(X_val, num_iteration=best_iter)
        oof_probs[val_mask.values] = preds

    hosp_df = hosp_df.copy()
    hosp_df["prob"] = oof_probs
    mean_best_iter = float(np.mean(best_iters))

    # Fit isotonic calibrator on OOF predictions
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(hosp_df["prob"].values, hosp_df["SepsisLabel"].values)
    hosp_df["cal_prob"] = calibrator.predict(hosp_df["prob"].values)

    # Search policy on calibrated OOF probabilities
    best_policy, _ = search_policy(hosp_df, "cal_prob", cfg)
    hosp_df["alert"] = apply_policy(
        hosp_df,
        "cal_prob",
        best_policy["threshold"],
        best_policy["k"],
        best_policy["sticky"],
    )

    metrics = evaluate_predictions(hosp_df, "cal_prob", "alert", cfg)
    return metrics, calibrator, best_policy, mean_best_iter, hosp_df


def run_hospital_external(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict[str, Any],
    mean_best_iter: float,
    calibrator: IsotonicRegression,
    policy: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train on all rows of source hospital and evaluate on target hospital.

    Returns:
        (metrics_dict, evaluated_test_df)
    """
    cat_features = ["static__unit"] if "static__unit" in feature_cols else []
    train_params = dict(params)
    train_params.pop("metric", None)
    train_params.pop("early_stopping_round", None)
    train_params.pop("early_stopping_rounds", None)
    train_params.pop("num_boost_round", None)
    train_params.update(
        {
            "seed": SEED,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
            "n_jobs": -1,
        }
    )

    num_round = max(50, round(1.1 * mean_best_iter))
    X_train = train_df[feature_cols]
    y_train = train_df["SepsisLabel"].astype(int)

    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
    booster = lgb.train(train_params, dtrain, num_boost_round=num_round)

    raw_preds = booster.predict(test_df[feature_cols])
    cal_preds = calibrator.predict(raw_preds)

    evaluated_test_df = test_df.copy()
    evaluated_test_df["prob"] = raw_preds
    evaluated_test_df["cal_prob"] = cal_preds
    evaluated_test_df["alert"] = apply_policy(
        evaluated_test_df,
        "cal_prob",
        policy["threshold"],
        policy["k"],
        policy["sticky"],
    )

    metrics = evaluate_predictions(evaluated_test_df, "cal_prob", "alert", cfg)
    return metrics, evaluated_test_df


def run_cross_hospital(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run full Phase 4 cross-hospital generalization experiments."""
    if cfg is None:
        cfg = load_config()
    set_seeds(SEED)

    tables_dir = Path(cfg["paths"]["tables"])
    figures_dir = Path(cfg["paths"]["figures"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load tuned parameters
    bp_path = Path("models/tuning/best_params.json")
    with open(bp_path) as f:
        tdata = json.load(f)
        best_params = tdata.get("best_params", tdata)

    lgbm_cfg = dict(cfg.get("lgbm", {}))
    lgbm_cfg.update(best_params)

    # Load standard features and feature list
    features_path = Path(cfg["paths"]["features"])
    logger.info("Loading feature matrix from %s...", features_path)
    features_df = pd.read_parquet(features_path)
    if "static__unit" in features_df.columns:
        features_df["static__unit"] = features_df["static__unit"].astype(int)

    with open(Path(cfg["paths"]["models"]) / "feature_list.json") as f:
        full_feature_cols = json.load(f)

    # Assign internal 5-fold CV within all patients of each hospital
    df_A = assign_hospital_folds(features_df, "A")
    df_B = assign_hospital_folds(features_df, "B")

    # Load or build R2 standardized features
    features_r2_df = get_standardized_features(cfg)
    if "static__unit" in features_r2_df.columns:
        features_r2_df["static__unit"] = features_r2_df["static__unit"].astype(int)
    df_A_r2 = assign_hospital_folds(features_r2_df, "A")
    df_B_r2 = assign_hospital_folds(features_r2_df, "B")

    # Define variants
    variants = {
        "R0": {
            "name": "R0 (Full)",
            "cols": full_feature_cols,
            "df_A": df_A,
            "df_B": df_B,
        },
        "R1": {
            "name": "R1 (No Missingness)",
            "cols": [
                c
                for c in full_feature_cols
                if not any(p in c for p in ["__meas", "__tsl", "__cnt24"])
            ],
            "df_A": df_A,
            "df_B": df_B,
        },
        "R2": {
            "name": "R2 (Per-Hospital Std)",
            "cols": full_feature_cols,
            "df_A": df_A_r2,
            "df_B": df_B_r2,
        },
        "R3": {
            "name": "R3 (No Admin Timing)",
            "cols": [
                c for c in full_feature_cols if c not in ["static__hosp_adm_time", "static__iculos"]
            ],
            "df_A": df_A,
            "df_B": df_B,
        },
    }

    results = []
    r0_data = {}

    for var_key, var_spec in variants.items():
        logger.info("==========================================")
        logger.info(
            "Executing Variant %s: %s (%d features)",
            var_key,
            var_spec["name"],
            len(var_spec["cols"]),
        )
        logger.info("==========================================")

        v_df_A = var_spec["df_A"]
        v_df_B = var_spec["df_B"]
        v_cols = var_spec["cols"]

        # 1. Hospital A internal reference
        logger.info("[%s] Training Hospital A Internal Reference (5-fold CV)...", var_key)
        m_A_int, cal_A, policy_A, iter_A, oof_A = run_hospital_internal_cv(
            v_df_A, v_cols, lgbm_cfg, cfg
        )
        logger.info(
            "[%s] A Internal: Utility=%.4f, AUROC=%.4f, AUPRC=%.4f (policy: %s)",
            var_key,
            m_A_int["utility_norm"],
            m_A_int["auroc"],
            m_A_int["auprc"],
            policy_A,
        )
        results.append({"variant": var_key, "direction": "A internal", **m_A_int})

        # 2. Hospital B internal reference
        logger.info("[%s] Training Hospital B Internal Reference (5-fold CV)...", var_key)
        m_B_int, cal_B, policy_B, iter_B, oof_B = run_hospital_internal_cv(
            v_df_B, v_cols, lgbm_cfg, cfg
        )
        logger.info(
            "[%s] B Internal: Utility=%.4f, AUROC=%.4f, AUPRC=%.4f (policy: %s)",
            var_key,
            m_B_int["utility_norm"],
            m_B_int["auroc"],
            m_B_int["auprc"],
            policy_B,
        )
        results.append({"variant": var_key, "direction": "B internal", **m_B_int})

        # 3. External test: A -> B
        logger.info("[%s] External Transfer: Train on Hospital A -> Test on Hospital B...", var_key)
        m_A_to_B, test_B = run_hospital_external(
            v_df_A, v_df_B, v_cols, lgbm_cfg, iter_A, cal_A, policy_A, cfg
        )
        logger.info(
            "[%s] A -> B: Utility=%.4f, AUROC=%.4f, AUPRC=%.4f",
            var_key,
            m_A_to_B["utility_norm"],
            m_A_to_B["auroc"],
            m_A_to_B["auprc"],
        )
        results.append({"variant": var_key, "direction": "A->B", **m_A_to_B})

        # 4. External test: B -> A
        logger.info("[%s] External Transfer: Train on Hospital B -> Test on Hospital A...", var_key)
        m_B_to_A, test_A = run_hospital_external(
            v_df_B, v_df_A, v_cols, lgbm_cfg, iter_B, cal_B, policy_B, cfg
        )
        logger.info(
            "[%s] B -> A: Utility=%.4f, AUROC=%.4f, AUPRC=%.4f",
            var_key,
            m_B_to_A["utility_norm"],
            m_B_to_A["auroc"],
            m_B_to_A["auprc"],
        )
        results.append({"variant": var_key, "direction": "B->A", **m_B_to_A})

        if var_key == "R0":
            r0_data = {
                "oof_A": oof_A,
                "oof_B": oof_B,
                "test_B": test_B,
                "test_A": test_A,
                "policy_A": policy_A,
                "policy_B": policy_B,
            }

    # Format output table
    res_df = pd.DataFrame(results)
    out_cols = [
        "variant",
        "direction",
        "utility_norm",
        "auroc",
        "auprc",
        "patient_sensitivity",
        "false_alerts_per_patient_day",
        "median_lead_time_h",
    ]
    # Filter to requested columns
    avail_cols = [c for c in out_cols if c in res_df.columns]
    res_df = res_df[avail_cols]
    csv_path = tables_dir / "cross_hospital.csv"
    res_df.to_csv(csv_path, index=False)
    logger.info("Cross-hospital results table saved to %s", csv_path)

    # ── Figure 1: Grouped Bars of Utility (Internal vs External per Variant) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    var_names = ["R0", "R1", "R2", "R3"]
    x = np.arange(len(var_names))
    width = 0.35

    # Subplot 1: Train on Hospital A
    a_int_vals = [
        res_df[(res_df["variant"] == v) & (res_df["direction"] == "A internal")][
            "utility_norm"
        ].values[0]
        for v in var_names
    ]
    a_ext_vals = [
        res_df[(res_df["variant"] == v) & (res_df["direction"] == "A->B")]["utility_norm"].values[0]
        for v in var_names
    ]
    axes[0].bar(
        x - width / 2,
        a_int_vals,
        width,
        label="A Internal (OOF)",
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0].bar(
        x + width / 2,
        a_ext_vals,
        width,
        label="A → B (External)",
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0].set_title("Source: Hospital A", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Normalized Utility", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["R0 (Full)", "R1 (No Miss)", "R2 (Per-Hosp Std)", "R3 (No Admin)"], fontsize=10
    )
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Train on Hospital B
    b_int_vals = [
        res_df[(res_df["variant"] == v) & (res_df["direction"] == "B internal")][
            "utility_norm"
        ].values[0]
        for v in var_names
    ]
    b_ext_vals = [
        res_df[(res_df["variant"] == v) & (res_df["direction"] == "B->A")]["utility_norm"].values[0]
        for v in var_names
    ]
    axes[1].bar(
        x - width / 2,
        b_int_vals,
        width,
        label="B Internal (OOF)",
        color="#3498db",
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1].bar(
        x + width / 2,
        b_ext_vals,
        width,
        label="B → A (External)",
        color="#e67e22",
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1].set_title("Source: Hospital B", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["R0 (Full)", "R1 (No Miss)", "R2 (Per-Hosp Std)", "R3 (No Admin)"], fontsize=10
    )
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        "Cross-Hospital Generalization: Internal Reference vs External Transfer",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig_path1 = figures_dir / "fig_cross_hospital.png"
    fig.savefig(fig_path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", fig_path1)

    # ── Figure 2: Score Distributions on Non-Septic Patients (Threshold Mismatch) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if r0_data:
        oof_A = r0_data["oof_A"]
        test_B = r0_data["test_B"]
        policy_A = r0_data["policy_A"]

        oof_B = r0_data["oof_B"]
        test_A = r0_data["test_A"]
        policy_B = r0_data["policy_B"]

        # Left panel: Model A applied to A non-septic vs B non-septic
        nonseptic_A_in_A = oof_A[oof_A["SepsisLabel"] == 0]["cal_prob"]
        nonseptic_B_in_B = test_B[test_B["SepsisLabel"] == 0]["cal_prob"]

        axes[0].hist(
            nonseptic_A_in_A,
            bins=50,
            range=(0, 0.4),
            density=True,
            alpha=0.5,
            color="#2ecc71",
            label="In Hosp A (Internal)",
        )
        axes[0].hist(
            nonseptic_B_in_B,
            bins=50,
            range=(0, 0.4),
            density=True,
            alpha=0.5,
            color="#e74c3c",
            label="In Hosp B (External)",
        )
        axes[0].axvline(
            policy_A["threshold"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Tau_A = {policy_A['threshold']:.2f}",
        )
        axes[0].set_title(
            "Model Trained on Hospital A\n(Predicted Risk on Non-Septic Patients)",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].set_xlabel("Calibrated Predicted Probability", fontsize=10)
        axes[0].set_ylabel("Density", fontsize=10)
        axes[0].legend(loc="upper right")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Right panel: Model B applied to B non-septic vs A non-septic
        nonseptic_B_in_B_int = oof_B[oof_B["SepsisLabel"] == 0]["cal_prob"]
        nonseptic_A_in_A_ext = test_A[test_A["SepsisLabel"] == 0]["cal_prob"]

        axes[1].hist(
            nonseptic_B_in_B_int,
            bins=50,
            range=(0, 0.4),
            density=True,
            alpha=0.5,
            color="#3498db",
            label="In Hosp B (Internal)",
        )
        axes[1].hist(
            nonseptic_A_in_A_ext,
            bins=50,
            range=(0, 0.4),
            density=True,
            alpha=0.5,
            color="#e67e22",
            label="In Hosp A (External)",
        )
        axes[1].axvline(
            policy_B["threshold"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Tau_B = {policy_B['threshold']:.2f}",
        )
        axes[1].set_title(
            "Model Trained on Hospital B\n(Predicted Risk on Non-Septic Patients)",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_xlabel("Calibrated Predicted Probability", fontsize=10)
        axes[1].set_ylabel("Density", fontsize=10)
        axes[1].legend(loc="upper right")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        "Cross-Hospital Domain Shift & Alert Threshold Mismatch",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig_path2 = figures_dir / "fig_score_distributions_A_vs_B.png"
    fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", fig_path2)

    return res_df


def main() -> None:
    """CLI entrypoint: python -m sepsis.cross_hospital."""
    parser = argparse.ArgumentParser(description="Cross-hospital generalization study")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    cfg = load_config(args.config)
    report = run_cross_hospital(cfg)
    print("\n=== Cross-Hospital Generalization Results ===")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
