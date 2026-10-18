"""Final LightGBM model trained on all dev patients — P3.T7."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import pandas as pd

from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED

logger = logging.getLogger(__name__)


def train_final_model(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    feature_cols: list,
    params: Dict[str, Any],
    mean_best_iter: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> lgb.Booster:
    """Train final model on all dev patients.

    Args:
        features_df: Full feature matrix.
        folds_df: Used to identify dev patients.
        feature_cols: Feature columns.
        params: LightGBM params (tuned).
        mean_best_iter: Mean best iteration from CV.
        cfg: Config dict.

    Returns:
        Trained booster.
    """
    if cfg is None:
        cfg = load_config()
    set_seeds(SEED)

    out_dir = Path(cfg["paths"]["models"]) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only dev patients
    dev_pids = set(folds_df["patient_id"])
    dev_df = features_df[features_df["patient_id"].isin(dev_pids)].copy()
    if "static__unit" in dev_df.columns:
        dev_df["static__unit"] = dev_df["static__unit"].astype(int)

    num_round = round(1.1 * mean_best_iter)
    logger.info(
        "Training final model on %d dev rows, %d rounds (1.1 × %.0f)",
        len(dev_df),
        num_round,
        mean_best_iter,
    )

    X = dev_df[feature_cols]
    y = dev_df["SepsisLabel"].astype(int)

    cat_features = ["static__unit"] if "static__unit" in feature_cols else []
    dtrain = lgb.Dataset(X, y, categorical_feature=cat_features, free_raw_data=False)

    # Remove early stopping for final model
    train_params = {k: v for k, v in params.items()}
    train_params.pop("metric", None)  # not needed without validation

    booster = lgb.train(
        train_params,
        dtrain,
        num_boost_round=num_round,
        callbacks=[lgb.log_evaluation(200)],
    )

    # Save model
    model_path = out_dir / "lgbm.txt"
    booster.save_model(str(model_path))
    logger.info("Final model saved to %s", model_path)

    # Save feature list and params
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(out_dir / "params.json", "w") as f:
        json.dump(
            {
                "params": params,
                "num_boost_round": num_round,
                "mean_best_iter": mean_best_iter,
            },
            f,
            indent=2,
        )

    return booster


def main() -> None:
    """CLI entrypoint: python -m sepsis.final_model."""
    parser = argparse.ArgumentParser(description="Train final LightGBM model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--tuned-dir",
        type=str,
        default="models/lgbm_tuned",
        help="Directory with tuned CV results (meta.json)",
    )
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    cfg = load_config(args.config)
    set_seeds(SEED)

    features_df = pd.read_parquet(cfg["paths"]["features"])
    folds_df = pd.read_csv(cfg["paths"]["folds"])

    with open(Path(cfg["paths"]["models"]) / "feature_list.json") as f:
        feature_cols = json.load(f)

    # Load tuned params
    tuned_dir = Path(args.tuned_dir)
    with open(tuned_dir / "meta.json") as f:
        meta = json.load(f)

    params = meta["params"]
    mean_best_iter = meta["mean_best_iter"]

    train_final_model(
        features_df,
        folds_df,
        feature_cols,
        params,
        mean_best_iter,
        cfg,
    )
    logger.info("Final model training complete.")


if __name__ == "__main__":
    main()
