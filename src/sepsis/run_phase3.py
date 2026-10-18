"""Phase 3 end-to-end runner:
features → baselines → LGBM default → tuning → tuned → ablations → final model.

Run: python -m sepsis.run_phase3 --config configs/config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from sepsis.baselines import run_all_baselines
from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED
from sepsis.features import build_features
from sepsis.final_model import train_final_model
from sepsis.train_lgbm import train_lgbm_cv
from sepsis.tune_lgbm import tune_lgbm

logger = logging.getLogger(__name__)


def run_phase3(
    cfg=None,
    skip_features=False,
    skip_baselines=False,
    skip_default=False,
    skip_tuning=False,
    skip_tuned=False,
    skip_ablations=False,
    skip_final=False,
):
    if cfg is None:
        cfg = load_config()
    set_seeds(SEED)

    models_dir = Path(cfg["paths"]["models"])
    features_path = Path(cfg["paths"]["features"])
    folds_path = Path(cfg["paths"]["folds"])
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── P3.T1: Build features ────────────────────────────────────────
    if not skip_features:
        logger.info("=== P3.T1: Building features ===")
        features_df = build_features(cfg=cfg)
    else:
        logger.info("Skipping feature build, loading from %s", features_path)
        features_df = pd.read_parquet(features_path)

    folds_df = pd.read_csv(folds_path)

    with open(models_dir / "feature_list.json") as f:
        feature_cols = json.load(f)

    logger.info("Features: %d cols, %d rows", len(feature_cols), len(features_df))

    # ── P3.T2: Run baselines ─────────────────────────────────────────
    if not skip_baselines:
        logger.info("=== P3.T2: Running baselines ===")
        run_all_baselines(features_df, folds_df, cfg)
    else:
        logger.info("Skipping baselines")

    # ── P3.T3: Default LGBM 5-fold ───────────────────────────────────
    if not skip_default:
        logger.info("=== P3.T3: Training LightGBM (default params) ===")
        oof_df_default, results_default = train_lgbm_cv(
            features_df,
            folds_df,
            feature_cols,
            cfg,
            output_dir="models/lgbm_default",
            experiment_name="lgbm_default",
        )
        default_metrics = results_default["oof_metrics"]
        default_mean_auprc = float(
            sum(fm["auprc"] for fm in results_default["fold_metrics"])
            / len(results_default["fold_metrics"])
        )
        logger.info(
            "Default OOF: utility=%.4f, AUROC=%.4f, AUPRC=%.4f",
            default_metrics["utility_norm"],
            default_metrics["auroc"],
            default_metrics["auprc"],
        )

        # Acceptance check
        if not (0.78 <= default_metrics["auroc"] <= 0.90):
            logger.warning(
                "⚠ AUROC %.4f is outside expected range [0.78, 0.90] — check for leakage or issues",
                default_metrics["auroc"],
            )
        if not (0.25 <= default_metrics["utility_norm"] <= 0.50):
            logger.warning(
                "⚠ utility_norm %.4f outside expected range [0.25, 0.50]",
                default_metrics["utility_norm"],
            )
    else:
        logger.info("Skipping default LGBM")
        with open("models/lgbm_default/meta.json") as f:
            meta = json.load(f)
        default_mean_auprc = float(
            sum(fm["auprc"] for fm in meta["fold_metrics"]) / len(meta["fold_metrics"])
        )
        results_default = {
            "oof_metrics": {},
            "fold_metrics": meta["fold_metrics"],
            "best_iters": meta["best_iterations"],
        }

    # ── P3.T4: Optuna tuning ─────────────────────────────────────────
    if not skip_tuning:
        logger.info("=== P3.T4: Optuna hyperparameter tuning ===")
        tune_result = tune_lgbm(features_df, folds_df, feature_cols, cfg)
        best_params = tune_result["best_params"]
        best_value = tune_result["best_value"]
        logger.info("Tuning best AUPRC: %.4f (default was %.4f)", best_value, default_mean_auprc)
        if best_value < default_mean_auprc:
            logger.warning("Tuned params underperform defaults — keeping default params")
            with open("models/lgbm_default/meta.json") as f:
                tuned_params_to_use = json.load(f)["params"]
        else:
            tuned_params_to_use = best_params
    else:
        logger.info("Skipping tuning, loading best params")
        bp_path = models_dir / "tuning" / "best_params.json"
        if bp_path.exists():
            with open(bp_path) as f:
                tune_result = json.load(f)
            best_params = tune_result["best_params"]
            tuned_params_to_use = best_params
        else:
            logger.warning("No tuning results found, using default params")
            with open("models/lgbm_default/meta.json") as f:
                tuned_params_to_use = json.load(f)["params"]

    # ── P3.T5: Tuned LGBM 5-fold ─────────────────────────────────────
    if not skip_tuned:
        logger.info("=== P3.T5: Training LightGBM (tuned params) ===")
        # Merge tuned params into cfg
        cfg_tuned = dict(cfg)
        cfg_tuned["lgbm"] = dict(cfg.get("lgbm", {}))
        cfg_tuned["lgbm"].update(tuned_params_to_use)

        oof_df_tuned, results_tuned = train_lgbm_cv(
            features_df,
            folds_df,
            feature_cols,
            cfg_tuned,
            output_dir="models/lgbm_tuned",
            experiment_name="lgbm_tuned",
        )
        tuned_metrics = results_tuned["oof_metrics"]
        logger.info(
            "Tuned OOF: utility=%.4f, AUROC=%.4f, AUPRC=%.4f",
            tuned_metrics["utility_norm"],
            tuned_metrics["auroc"],
            tuned_metrics["auprc"],
        )

        # Comparison table
        if not skip_default:
            comparison = pd.DataFrame(
                [
                    {"model": "default", **results_default.get("oof_metrics", {})},
                    {"model": "tuned", **tuned_metrics},
                ]
            )
            comparison.to_csv(tables_dir / "lgbm_dev_oof.csv", index=False)
            logger.info("Comparison table saved to %s", tables_dir / "lgbm_dev_oof.csv")
    else:
        logger.info("Skipping tuned LGBM")
        with open("models/lgbm_tuned/meta.json") as f:
            results_tuned = json.load(f)

    # ── P3.T6: Ablations (optional — slow) ───────────────────────────
    if not skip_ablations:
        logger.info("=== P3.T6: Feature-group ablations ===")
        from sepsis.ablations import run_ablations

        run_ablations(features_df, folds_df, feature_cols, cfg)
    else:
        logger.info("Skipping ablations")

    # ── P3.T7: Final model ───────────────────────────────────────────
    if not skip_final:
        logger.info("=== P3.T7: Training final model ===")
        with open("models/lgbm_tuned/meta.json") as f:
            tuned_meta = json.load(f)
        final_params = tuned_meta["params"]
        mean_best_iter = tuned_meta["mean_best_iter"]

        train_final_model(features_df, folds_df, feature_cols, final_params, mean_best_iter, cfg)
        logger.info("Phase 3 complete. Final model saved to models/final/")
    else:
        logger.info("Skipping final model")

    logger.info("=== Phase 3 complete ===")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 end-to-end")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-default", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-tuned", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    cfg = load_config(args.config)
    run_phase3(
        cfg=cfg,
        skip_features=args.skip_features,
        skip_baselines=args.skip_baselines,
        skip_default=args.skip_default,
        skip_tuning=args.skip_tuning,
        skip_tuned=args.skip_tuned,
        skip_ablations=args.skip_ablations,
        skip_final=args.skip_final,
    )


if __name__ == "__main__":
    main()
