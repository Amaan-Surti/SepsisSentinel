"""LightGBM 5-fold dev training for SepsisSentinel — P3.T3 / P3.T5."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from sepsis.alerting import apply_policy, search_policy
from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED
from sepsis.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


def train_lgbm_cv(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    feature_cols: list,
    cfg: Optional[Dict[str, Any]] = None,
    output_dir: str = "models/lgbm_default",
    experiment_name: str = "lgbm_default",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Train LightGBM with 5-fold CV and collect OOF predictions.

    Args:
        features_df: Full feature matrix.
        folds_df: Fold assignments (patient_id, fold).
        feature_cols: Ordered list of feature column names.
        cfg: Config dict.
        output_dir: Directory to save models and OOF predictions.
        experiment_name: MLflow experiment name suffix.

    Returns:
        (oof_df, results_dict)
    """
    if cfg is None:
        cfg = load_config()
    set_seeds(cfg.get("seed", SEED))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge fold info
    fold_map = folds_df.set_index("patient_id")["fold"].to_dict()
    features_df = features_df.copy()
    features_df["fold"] = features_df["patient_id"].map(fold_map)
    dev_df = features_df[features_df["fold"].notna()].copy()
    dev_df["fold"] = dev_df["fold"].astype(int)
    if "static__unit" in dev_df.columns:
        dev_df["static__unit"] = dev_df["static__unit"].astype(int)

    # LightGBM params
    lgbm_cfg = cfg.get("lgbm", {})
    params = {
        "objective": lgbm_cfg.get("objective", "binary"),
        "learning_rate": lgbm_cfg.get("learning_rate", 0.03),
        "num_leaves": lgbm_cfg.get("num_leaves", 63),
        "max_depth": lgbm_cfg.get("max_depth", -1),
        "min_child_samples": lgbm_cfg.get("min_child_samples", 200),
        "feature_fraction": lgbm_cfg.get("feature_fraction", 0.7),
        "bagging_fraction": lgbm_cfg.get("bagging_fraction", 0.8),
        "bagging_freq": lgbm_cfg.get("bagging_freq", 1),
        "lambda_l1": lgbm_cfg.get("lambda_l1", 0.0),
        "lambda_l2": lgbm_cfg.get("lambda_l2", 1.0),
        "scale_pos_weight": lgbm_cfg.get("scale_pos_weight", 5.0),
        "max_bin": lgbm_cfg.get("max_bin", 255),
        "metric": lgbm_cfg.get("metric", "average_precision"),
        "verbose": lgbm_cfg.get("verbose", -1),
        "n_jobs": lgbm_cfg.get("n_jobs", -1),
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
    }
    num_boost_round = lgbm_cfg.get("num_boost_round", 5000)
    early_stopping_rounds = lgbm_cfg.get("early_stopping_rounds", 200)

    # Categorical feature
    cat_features = ["static__unit"] if "static__unit" in feature_cols else []

    n_folds = dev_df["fold"].nunique()
    oof_probs = np.full(len(dev_df), np.nan)
    best_iters = []
    fold_metrics = []

    for fold_idx in range(n_folds):
        logger.info("=== Fold %d / %d ===", fold_idx, n_folds - 1)
        train_mask = dev_df["fold"] != fold_idx
        val_mask = dev_df["fold"] == fold_idx

        X_train = dev_df.loc[train_mask, feature_cols]
        y_train = dev_df.loc[train_mask, "SepsisLabel"].astype(int)
        X_val = dev_df.loc[val_mask, feature_cols]
        y_val = dev_df.loc[val_mask, "SepsisLabel"].astype(int)

        dtrain = lgb.Dataset(
            X_train,
            y_train,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            y_val,
            categorical_feature=cat_features,
            free_raw_data=False,
            reference=dtrain,
        )

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(100),
            ],
        )

        best_iter = booster.best_iteration
        best_iters.append(best_iter)
        logger.info("Fold %d best iteration: %d", fold_idx, best_iter)

        # Save model
        model_path = out_dir / f"fold{fold_idx}.txt"
        booster.save_model(str(model_path))

        # OOF predictions
        preds = booster.predict(X_val, num_iteration=best_iter)
        oof_probs[val_mask.values] = preds

        # Fold-level AUROC/AUPRC
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = roc_auc_score(y_val, preds)
        auprc = average_precision_score(y_val, preds)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "best_iter": best_iter,
                "auroc": auroc,
                "auprc": auprc,
            }
        )
        logger.info("Fold %d: AUROC=%.4f, AUPRC=%.4f", fold_idx, auroc, auprc)

    # Save OOF
    dev_df["prob"] = oof_probs
    oof_df = dev_df[["patient_id", "hour", "SepsisLabel", "prob"]].copy()
    oof_path = out_dir / "oof.parquet"
    oof_df.to_parquet(oof_path, index=False)
    logger.info("OOF predictions saved to %s", oof_path)

    # Save best iterations
    meta = {
        "params": params,
        "best_iterations": best_iters,
        "mean_best_iter": float(np.mean(best_iters)),
        "fold_metrics": fold_metrics,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # MLflow logging (optional — skip if not installed)
    try:
        import mlflow

        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("sepsis")
        with mlflow.start_run(run_name=experiment_name):
            mlflow.log_params({k: v for k, v in params.items() if not isinstance(v, bool)})
            for fm in fold_metrics:
                mlflow.log_metric(f"fold{fm['fold']}_auroc", fm["auroc"])
                mlflow.log_metric(f"fold{fm['fold']}_auprc", fm["auprc"])
                mlflow.log_metric(f"fold{fm['fold']}_best_iter", fm["best_iter"])
            mlflow.log_metric("mean_best_iter", float(np.mean(best_iters)))
        logger.info("MLflow run logged to experiment 'sepsis'")
    except ImportError:
        logger.warning("MLflow not installed, skipping logging")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    # Search policy on OOF
    logger.info("Searching alert policy on OOF predictions...")
    oof_with_label = dev_df[["patient_id", "hour", "SepsisLabel", "prob"]].copy()
    best_params, grid_df = search_policy(oof_with_label, "prob", cfg)
    oof_with_label["alert"] = apply_policy(
        oof_with_label,
        "prob",
        best_params["threshold"],
        best_params["k"],
        best_params["sticky"],
    )
    oof_metrics = evaluate_predictions(oof_with_label, "prob", "alert", cfg)
    oof_metrics.update(best_params)

    logger.info(
        "OOF metrics: utility_norm=%.4f, AUROC=%.4f, AUPRC=%.4f",
        oof_metrics["utility_norm"],
        oof_metrics["auroc"],
        oof_metrics["auprc"],
    )

    # Save grid
    grid_path = Path(cfg["paths"]["tables"]) / f"policy_grid_{experiment_name}.csv"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(grid_path, index=False)

    results = {
        "oof_metrics": oof_metrics,
        "fold_metrics": fold_metrics,
        "best_iters": best_iters,
        "best_policy": best_params,
    }

    return oof_df, results


def main() -> None:
    """CLI entrypoint: python -m sepsis.train_lgbm."""
    parser = argparse.ArgumentParser(description="Train LightGBM 5-fold CV")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output-dir", type=str, default="models/lgbm_default")
    parser.add_argument("--name", type=str, default="lgbm_default")
    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Optional JSON file with tuned params to override config",
    )
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    cfg = load_config(args.config)
    set_seeds(SEED)

    # Check for tuned params
    if args.params_file and Path(args.params_file).exists():
        with open(args.params_file) as f:
            tdata = json.load(f)
            bp = tdata.get("best_params", tdata)
            cfg.setdefault("lgbm", {}).update(bp)
            logger.info("Loaded tuned parameters from %s: %s", args.params_file, bp)
    elif args.name == "lgbm_tuned":
        bp_path = Path(cfg["paths"]["models"]) / "tuning" / "best_params.json"
        if bp_path.exists():
            with open(bp_path) as f:
                tdata = json.load(f)
                bp = tdata.get("best_params", tdata)
                cfg.setdefault("lgbm", {}).update(bp)
                logger.info("Auto-loaded tuned parameters from %s: %s", bp_path, bp)

    features_df = pd.read_parquet(cfg["paths"]["features"])
    folds_df = pd.read_csv(cfg["paths"]["folds"])

    with open(Path(cfg["paths"]["models"]) / "feature_list.json") as f:
        feature_cols = json.load(f)

    oof_df, results = train_lgbm_cv(
        features_df,
        folds_df,
        feature_cols,
        cfg,
        output_dir=args.output_dir,
        experiment_name=args.name,
    )

    # Save metrics to reports/tables/lgbm_dev_oof.csv
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    oof_csv_path = tables_dir / "lgbm_dev_oof.csv"
    row = {"model": args.name, **results["oof_metrics"]}
    if oof_csv_path.exists():
        existing_df = pd.read_csv(oof_csv_path)
        existing_df = existing_df[existing_df["model"] != args.name]
        combined = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    combined.to_csv(oof_csv_path, index=False)

    # Print summary
    print("\n=== OOF Summary ===")
    for k, v in results["oof_metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
