"""Feature-group ablation study — P3.T6.

Run LightGBM with tuned params, each time removing one feature group.
Compares utility/AUROC/AUPRC to the full model.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED
from sepsis.train_lgbm import train_lgbm_cv

logger = logging.getLogger(__name__)

# Ablation groups per §7.7
ABLATION_GROUPS = {
    "no_missingness": {
        "description": "Remove F2+F3+F4 (missingness signals)",
        "patterns": ["__meas", "__tsl", "__cnt24"],
    },
    "no_rolling": {
        "description": "Remove F5 (rolling stats)",
        "patterns": ["__mean_", "__min_", "__max_", "__std_", "__slope_"],
    },
    "no_deltas": {
        "description": "Remove F6+F7 (diffs + baseline deltas)",
        "patterns": ["__diff6", "__dbase"],
    },
    "no_indices_scores": {
        "description": "Remove F8+F9 (indices + scores)",
        "patterns": ["idx__", "score__"],
    },
}


def _filter_features(feature_cols: List[str], patterns: List[str]) -> List[str]:
    """Return feature cols with those matching any pattern removed."""
    return [c for c in feature_cols if not any(p in c for p in patterns)]


def run_ablations(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run ablation experiments and produce comparison table + figure.

    Args:
        features_df: Full feature matrix.
        folds_df: Fold assignments.
        feature_cols: Full feature list.
        cfg: Config dict.

    Returns:
        DataFrame with ablation results.
    """
    if cfg is None:
        cfg = load_config()
    cfg = dict(cfg)
    best_params_path = Path("models/tuning/best_params.json")
    if best_params_path.exists():
        with open(best_params_path) as f:
            tdata = json.load(f)
            best_params = tdata.get("best_params", tdata)
            cfg.setdefault("lgbm", {}).update(best_params)
            logger.info("Loaded tuned hyperparameters for ablations: %s", best_params)

    set_seeds(SEED)

    tables_dir = Path(cfg["paths"]["tables"])
    figures_dir = Path(cfg["paths"]["figures"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Full model (reference)
    tuned_oof_csv = tables_dir / "lgbm_dev_oof.csv"
    if (
        Path("models/lgbm_tuned/oof.parquet").exists()
        and tuned_oof_csv.exists()
        and "lgbm_tuned" in pd.read_csv(tuned_oof_csv)["model"].values
    ):
        logger.info("Reusing existing tuned model metrics for reference...")
        m_df = pd.read_csv(tuned_oof_csv)
        row = m_df[m_df["model"] == "lgbm_tuned"].iloc[0]
        full_m = {
            "utility_norm": float(row["utility_norm"]),
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
        }
    else:
        logger.info("Running full model (reference)...")
        _, full_res = train_lgbm_cv(
            features_df,
            folds_df,
            feature_cols,
            cfg,
            output_dir="models/ablation_full",
            experiment_name="ablation_full",
        )
        full_m = full_res["oof_metrics"]
    results.append(
        {
            "variant": "full",
            "description": "Full feature set",
            "n_features": len(feature_cols),
            "utility_norm": full_m["utility_norm"],
            "auroc": full_m["auroc"],
            "auprc": full_m["auprc"],
        }
    )

    # Each ablation
    for name, spec in ABLATION_GROUPS.items():
        ablated_cols = _filter_features(feature_cols, spec["patterns"])
        n_removed = len(feature_cols) - len(ablated_cols)
        logger.info(
            "Running ablation '%s': removing %d features (%s)...",
            name,
            n_removed,
            spec["description"],
        )

        _, abl_res = train_lgbm_cv(
            features_df,
            folds_df,
            ablated_cols,
            cfg,
            output_dir=f"models/ablation_{name}",
            experiment_name=f"ablation_{name}",
        )
        abl_m = abl_res["oof_metrics"]
        results.append(
            {
                "variant": name,
                "description": spec["description"],
                "n_features": len(ablated_cols),
                "utility_norm": abl_m["utility_norm"],
                "auroc": abl_m["auroc"],
                "auprc": abl_m["auprc"],
            }
        )

    report = pd.DataFrame(results)
    report.to_csv(tables_dir / "ablations.csv", index=False)
    logger.info("Ablation results saved to %s", tables_dir / "ablations.csv")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    variants = report["variant"].values
    x = np.arange(len(variants))

    for ax, metric in zip(axes, ["utility_norm", "auroc", "auprc"]):
        vals = report[metric].values
        colors = ["#2ecc71" if v == "full" else "#e74c3c" for v in variants]
        ax.barh(x, vals, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(x)
        ax.set_yticklabels(variants, fontsize=9)
        ax.set_xlabel(metric)
        ax.set_title(metric.upper())
        # Add reference line for full model
        ax.axvline(vals[0], color="#2ecc71", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(figures_dir / "fig_ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ablation figure saved to %s", figures_dir / "fig_ablations.png")

    return report


def main() -> None:
    """CLI entrypoint: python -m sepsis.ablations."""
    parser = argparse.ArgumentParser(description="Feature-group ablation study")
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

    report = run_ablations(features_df, folds_df, feature_cols, cfg)
    print("\nAblation Results:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
