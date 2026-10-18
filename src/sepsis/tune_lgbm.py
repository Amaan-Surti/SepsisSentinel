"""Optuna hyperparameter tuning for LightGBM — P3.T4."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED

logger = logging.getLogger(__name__)


def _objective(
    trial: optuna.Trial,
    cached_splits: list,
    cat_features: list,
    cfg: Dict[str, Any],
) -> float:
    """Single Optuna trial: 3-fold CV, return mean val AUPRC."""
    # Suggest hyperparameters per §7.5
    lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    num_leaves = trial.suggest_int("num_leaves", 15, 255, log=True)
    min_child = trial.suggest_int("min_child_samples", 20, 1000, log=True)
    feat_frac = trial.suggest_float("feature_fraction", 0.4, 1.0)
    bag_frac = trial.suggest_float("bagging_fraction", 0.5, 1.0)
    l2 = trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True)
    spw = trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)

    lgbm_cfg = cfg.get("lgbm", {})
    params = {
        "objective": "binary",
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "max_depth": lgbm_cfg.get("max_depth", -1),
        "min_child_samples": min_child,
        "feature_fraction": feat_frac,
        "bagging_fraction": bag_frac,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": l2,
        "scale_pos_weight": spw,
        "max_bin": lgbm_cfg.get("max_bin", 255),
        "metric": "average_precision",
        "verbose": -1,
        "n_jobs": lgbm_cfg.get("n_jobs", -1),
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
    }

    auprcs = []
    for X_train, y_train, X_val, y_val in cached_splits:
        dtrain = lgb.Dataset(
            X_train, y_train, categorical_feature=cat_features, free_raw_data=False
        )
        dval = lgb.Dataset(
            X_val, y_val, categorical_feature=cat_features, free_raw_data=False, reference=dtrain
        )

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg["lgbm"].get("num_boost_round", 5000),
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(cfg["lgbm"].get("early_stopping_rounds", 200)),
                lgb.log_evaluation(0),  # silent
            ],
        )

        preds = booster.predict(X_val, num_iteration=booster.best_iteration)
        auprc = average_precision_score(y_val, preds)
        auprcs.append(auprc)

    mean_auprc = float(np.mean(auprcs))
    logger.info("Trial %d: mean AUPRC=%.4f (folds %s)", trial.number, mean_auprc, auprcs)
    return mean_auprc


def tune_lgbm(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    feature_cols: list,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Optuna TPE search for LightGBM hyperparameters.

    Returns:
        Dict with best_params and best_value.
    """
    if cfg is None:
        cfg = load_config()
    set_seeds(SEED)

    tuning_cfg = cfg.get("tuning", {})
    n_trials = tuning_cfg.get("n_trials", 40)
    timeout_hours = tuning_cfg.get("timeout_hours", 3)
    folds_used = tuning_cfg.get("folds_used", [0, 1, 2])

    out_dir = Path(cfg["paths"]["models"]) / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge fold info
    fold_map = folds_df.set_index("patient_id")["fold"].to_dict()
    dev_df = features_df[features_df["patient_id"].isin(fold_map)].copy()
    dev_df["fold"] = dev_df["patient_id"].map(fold_map).astype(int)
    if "static__unit" in dev_df.columns:
        dev_df["static__unit"] = dev_df["static__unit"].astype(int)

    cat_features = ["static__unit"] if "static__unit" in feature_cols else []

    logger.info("Pre-splitting data for tuning folds %s...", folds_used)
    cached_splits = []
    for val_fold in folds_used:
        train_mask = dev_df["fold"] != val_fold
        val_mask = dev_df["fold"] == val_fold

        train_sub = dev_df[train_mask]
        # Low-RAM option: keep all rows of septic patients and random 50% of non-septic in training
        sep_mask = train_sub["SepsisLabel"] == 1
        nonsep = train_sub[~sep_mask]
        sep = train_sub[sep_mask]
        nonsep_sample = nonsep.sample(frac=0.5, random_state=SEED)
        train_sub = pd.concat([sep, nonsep_sample])

        X_train = train_sub[feature_cols].copy()
        y_train = train_sub["SepsisLabel"].astype(int).values
        X_val = dev_df.loc[val_mask, feature_cols].copy()
        y_val = dev_df.loc[val_mask, "SepsisLabel"].astype(int).values
        cached_splits.append((X_train, y_train, X_val, y_val))
    logger.info("Pre-splitting complete: %d folds cached.", len(cached_splits))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.enqueue_trial(
        {
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 200,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "lambda_l2": 1.0,
            "scale_pos_weight": 5.0,
        }
    )

    study.optimize(
        lambda trial: _objective(trial, cached_splits, cat_features, cfg),
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value = study.best_value
    logger.info("Best params: %s", best_params)
    logger.info("Best AUPRC: %.4f", best_value)

    # Save best params
    with open(out_dir / "best_params.json", "w") as f:
        json.dump({"best_params": best_params, "best_value": best_value}, f, indent=2)

    # Save trials CSV
    trials_df = study.trials_dataframe()
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(tables_dir / "optuna_trials.csv", index=False)
    logger.info("Trials saved to %s", tables_dir / "optuna_trials.csv")

    return {"best_params": best_params, "best_value": best_value}


def main() -> None:
    """CLI entrypoint: python -m sepsis.tune_lgbm."""
    parser = argparse.ArgumentParser(description="Optuna tuning for LightGBM")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    cfg = load_config(args.config)
    set_seeds(SEED)

    features_df = pd.read_parquet(cfg["paths"]["features"])
    folds_df = pd.read_csv(cfg["paths"]["folds"])

    with open(Path(cfg["paths"]["models"]) / "feature_list.json") as f:
        feature_cols = json.load(f)

    result = tune_lgbm(features_df, folds_df, feature_cols, cfg)
    print(f"\nBest AUPRC: {result['best_value']:.4f}")
    print(f"Best params: {result['best_params']}")


if __name__ == "__main__":
    main()
