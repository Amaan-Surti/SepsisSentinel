"""Explainability module for SepsisSentinel using SHAP — P5.T3.

Provides clinical feature-to-text translation, TreeExplainer extraction,
global beeswarm and bar visualizations, local row explanations, and fold stability checks.
"""

import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VAR_LABELS = {
    "HR": "Heart rate",
    "O2Sat": "Oxygen saturation",
    "Temp": "Temperature",
    "SBP": "Systolic BP",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic BP",
    "Resp": "Respiratory rate",
    "EtCO2": "End-tidal CO2",
    "Lactate": "Lactate",
    "WBC": "White cell count",
    "Creatinine": "Creatinine",
    "Platelets": "Platelets",
    "BUN": "Urea (BUN)",
    "Bilirubin_total": "Bilirubin",
    "pH": "Blood pH",
    "HCO3": "Bicarbonate",
    "BaseExcess": "Base excess",
    "FiO2": "FiO2",
    "PaCO2": "PaCO2",
    "SaO2": "SaO2",
    "AST": "AST",
    "Alkalinephos": "Alkaline phosphatase",
    "Calcium": "Calcium",
    "Chloride": "Chloride",
    "Bilirubin_direct": "Direct bilirubin",
    "Glucose": "Glucose",
    "Magnesium": "Magnesium",
    "Phosphate": "Phosphate",
    "Potassium": "Potassium",
    "TroponinI": "Troponin I",
    "Hct": "Hematocrit",
    "Hgb": "Hemoglobin",
    "PTT": "Partial thromboplastin time",
    "Fibrinogen": "Fibrinogen",
}


def feature_text(name: str, value: float) -> str:
    """Translate a feature name and numeric value into clinical, human-readable text.

    Follows SepsisSentinel Build Plan §9.3 conventions.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        base_name = name.split("__")[0]
        label = VAR_LABELS.get(base_name, base_name)
        return f"{label} not available"

    # 1. Static demographic and administrative features
    if name == "static__age":
        return f"Age {value:.0f} years"
    if name == "static__gender":
        return "Male" if int(round(value)) == 1 else "Female"
    if name == "static__unit":
        u = int(round(value))
        return "Unit: " + ("MICU" if u == 1 else "SICU" if u == 2 else "unknown ICU")
    if name == "static__hosp_adm_time":
        if value <= 0:
            return f"Hospital stay {abs(value):.1f}h prior to ICU"
        return f"Hospital admission {value:+.1f}h relative to ICU"
    if name == "static__iculos":
        return f"ICU stay hour {value:.0f}"

    # 2. Clinical indices and composite scores
    if name == "idx__shock_index__slope_6h":
        direction = "rising" if value > 0 else "falling"
        return f"Shock index {direction} over 6h"
    if name == "idx__shock_index":
        return f"Shock index {value:.2f}"
    if name == "idx__mod_shock_index":
        return f"Modified shock index {value:.2f}"
    if name == "idx__pulse_pressure":
        return f"Pulse pressure {value:.1f} mmHg"
    if name == "idx__bun_cr":
        return f"BUN/Creatinine ratio {value:.1f}"
    if name == "idx__sf_ratio":
        return f"SpO2/FiO2 ratio {value:.1f}"
    if name == "score__sirs":
        return f"SIRS criteria met: {value:.0f}/4"
    if name == "score__qsofa_partial":
        return f"Partial qSOFA: {value:.0f}/2 criteria met"
    if name == "score__sofa_partial_d24":
        return f"Partial SOFA change: {value:+.0f} in 24h"
    if name == "score__sofa_partial":
        return f"Partial SOFA score {value:.0f}/13"

    # Split for measurements, rolling stats, deltas
    parts = name.split("__")
    raw_var = parts[0]
    label = VAR_LABELS.get(raw_var, raw_var)

    if len(parts) > 1:
        suffix = parts[1]

        # Missingness & sampling frequency
        if suffix == "meas":
            return (
                f"{label} measured this hour"
                if int(round(value)) == 1
                else f"{label} not measured this hour"
            )
        if suffix == "tsl":
            return f"{label} last measured {value:.0f}h ago"
        if suffix == "cnt24":
            return f"{label} tested {value:.0f}× in 24h"

        # Deltas
        if suffix == "diff6":
            return f"{label} changed {value:+.1f} in 6h"
        if suffix == "dbase":
            return f"{label} {value:+.1f} vs patient baseline"

        # Rolling statistics
        # Pattern e.g. slope_6h, mean_6h, min_12h, max_24h, std_6h
        if suffix.startswith("slope_"):
            hours = suffix.split("_")[1].replace("h", "")
            direction = "rising" if value > 0 else "falling"
            return f"{label} {direction} over {hours}h"
        if suffix.startswith("mean_"):
            hours = suffix.split("_")[1].replace("h", "")
            return f"{label} mean over {hours}h: {value:.1f}"
        if suffix.startswith("min_"):
            hours = suffix.split("_")[1].replace("h", "")
            return f"{label} min over {hours}h: {value:.1f}"
        if suffix.startswith("max_"):
            hours = suffix.split("_")[1].replace("h", "")
            return f"{label} max over {hours}h: {value:.1f}"
        if suffix.startswith("std_"):
            hours = suffix.split("_")[1].replace("h", "")
            return f"{label} std over {hours}h: {value:.1f}"

    # Default fallback for current / LOCF values
    return f"{label} {value:.1f}"


def explain_rows(
    explainer: shap.TreeExplainer,
    X_rows: pd.DataFrame,
    top_n: int = 5,
) -> List[List[Dict[str, Any]]]:
    """Generate top positive SHAP explanation dicts for each row."""
    feature_names = list(X_rows.columns)
    shap_values = explainer.shap_values(X_rows)

    # For binary classification, shap_values can be a list [class0, class1] or ndarray
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    elif shap_values.ndim == 3:  # (N, features, classes)
        shap_matrix = shap_values[:, :, 1]
    else:
        shap_matrix = shap_values

    explanations = []
    for i in range(len(X_rows)):
        row_shaps = shap_matrix[i]
        row_vals = X_rows.iloc[i].values

        pos_items = []
        for feat, val, s in zip(feature_names, row_vals, row_shaps):
            if s > 0:  # only positive contributions pushing toward alert
                pos_items.append(
                    {
                        "feature": feat,
                        "value": float(val) if not pd.isna(val) else None,
                        "shap": float(s),
                        "text": feature_text(feat, val),
                    }
                )

        # Sort descending by shap
        pos_items.sort(key=lambda x: x["shap"], reverse=True)
        explanations.append(pos_items[:top_n])

    return explanations


def sample_dev_cohort(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    n_septic_pts: int = 300,
    n_nonseptic_rows: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample dev rows for SHAP global analysis per Build Plan §9.3.

    All rows of 300 random septic dev patients + 20,000 random non-septic dev rows.
    """
    dev_pids = set(folds_df["patient_id"])
    dev_mask = features_df["patient_id"].isin(dev_pids)
    dev_data = features_df[dev_mask]

    # Find septic patients
    septic_pids = sorted(list(dev_data[dev_data["SepsisLabel"] == 1]["patient_id"].unique()))
    rng = np.random.RandomState(seed)
    chosen_septic_pts = rng.choice(
        septic_pids, size=min(n_septic_pts, len(septic_pids)), replace=False
    )

    septic_rows = dev_data[dev_data["patient_id"].isin(chosen_septic_pts)]

    # Non-septic dev rows (from patients with no SepsisLabel==1)
    nonseptic_dev_mask = ~dev_data["patient_id"].isin(set(septic_pids))
    nonseptic_rows = dev_data[nonseptic_dev_mask]

    n_sample_ns = min(n_nonseptic_rows, len(nonseptic_rows))
    sampled_ns_rows = nonseptic_rows.sample(n=n_sample_ns, random_state=seed)

    combined = pd.concat([septic_rows, sampled_ns_rows], axis=0).reset_index(drop=True)
    logger.info(
        "SHAP sample: %d septic rows (from %d patients) + %d non-septic rows = %d total rows",
        len(septic_rows),
        len(chosen_septic_pts),
        len(sampled_ns_rows),
        len(combined),
    )
    return combined


def compute_shap_global(
    booster: lgb.Booster,
    sample_df: pd.DataFrame,
    feature_list: List[str],
) -> Tuple[np.ndarray, shap.TreeExplainer, pd.DataFrame]:
    """Compute SHAP values and global feature importance on sample."""
    X = sample_df[feature_list]
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    elif shap_values.ndim == 3:
        shap_matrix = shap_values[:, :, 1]
    else:
        shap_matrix = shap_values

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    df_global = (
        pd.DataFrame(
            {
                "feature": feature_list,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return shap_matrix, explainer, df_global


def plot_shap_figures(
    shap_matrix: np.ndarray,
    X_sample: pd.DataFrame,
    df_global: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Plot beeswarm and bar charts for top 20 features."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar plot top 20
    top20 = df_global.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top20))[::-1]
    labels = [f"{f} ({feature_text(f, 0.0).split(':')[0]})" for f in top20["feature"]]
    ax.barh(y_pos, top20["mean_abs_shap"], color="#0E7C7B", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (Average Impact on Prediction)", fontsize=11)
    ax.set_title("Top 20 Features by Mean |SHAP| (Global Impact)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    plt.tight_layout()
    bar_path = out_dir / "fig_shap_bar_top20.png"
    fig.savefig(bar_path, dpi=300)
    plt.close(fig)
    logger.info("Saved SHAP bar plot to %s", bar_path)

    # 2. Beeswarm plot top 20
    top20_feats = list(top20["feature"])
    feat_indices = [X_sample.columns.get_loc(f) for f in top20_feats]
    shap_top20 = shap_matrix[:, feat_indices]
    X_top20 = X_sample[top20_feats]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_top20,
        X_top20,
        feature_names=top20_feats,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Beeswarm Plot: Top 20 Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = out_dir / "fig_shap_beeswarm_top20.png"
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    logger.info("Saved SHAP beeswarm plot to %s", beeswarm_path)


def compute_stability(
    sample_df: pd.DataFrame,
    feature_list: List[str],
    models_dir: Path,
) -> pd.DataFrame:
    """Compute top-10 feature stability across 5 folds and report pairwise Jaccard index."""
    X = sample_df[feature_list].iloc[:5000]  # Fast subset for fold stability
    fold_top10 = {}

    for fold in range(5):
        fold_model_path = models_dir / f"fold{fold}.txt"
        if not fold_model_path.exists():
            continue
        booster = lgb.Booster(model_file=str(fold_model_path))
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sm = sv[1]
        elif sv.ndim == 3:
            sm = sv[:, :, 1]
        else:
            sm = sv
        mean_abs = np.mean(np.abs(sm), axis=0)
        top10_idx = np.argsort(mean_abs)[::-1][:10]
        fold_top10[fold] = set(feature_list[i] for i in top10_idx)
        logger.info("Fold %d top-10 SHAP features: %s", fold, fold_top10[fold])

    # Pairwise Jaccard
    jaccards = []
    pairs = list(itertools.combinations(fold_top10.keys(), 2))
    for f1, f2 in pairs:
        s1 = fold_top10[f1]
        s2 = fold_top10[f2]
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        j = intersection / union if union > 0 else 1.0
        jaccards.append({"fold_1": f1, "fold_2": f2, "jaccard": j})

    df_jaccard = pd.DataFrame(jaccards)
    mean_j = float(df_jaccard["jaccard"].mean()) if len(df_jaccard) > 0 else 1.0
    logger.info("Mean pairwise Jaccard across fold models: %.4f", mean_j)
    return df_jaccard


def run_explain() -> None:
    """Run full SHAP explainability workflow."""
    with open("models/final/feature_list.json") as f:
        feature_list = json.load(f)

    final_model_path = Path("models/final/lgbm.txt")
    if not final_model_path.exists():
        raise FileNotFoundError(f"Missing {final_model_path}")

    booster = lgb.Booster(model_file=str(final_model_path))

    # Load splits and features
    folds_df = pd.read_csv("data/splits/folds.csv")
    features_df = pd.read_parquet("data/processed/features.parquet")

    # Sample cohort
    sample_df = sample_dev_cohort(features_df, folds_df, n_septic_pts=300, n_nonseptic_rows=20000)

    # Global SHAP
    shap_matrix, explainer, df_global = compute_shap_global(booster, sample_df, feature_list)

    # Save shap_global.csv
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    shap_csv_path = tables_dir / "shap_global.csv"
    df_global.to_csv(shap_csv_path, index=False)
    logger.info("Saved SHAP global table to %s", shap_csv_path)

    # Plot figures
    figs_dir = Path("reports/figures")
    plot_shap_figures(shap_matrix, sample_df[feature_list], df_global, figs_dir)

    # Fold stability check
    stability_df = compute_stability(sample_df, feature_list, Path("models/lgbm_tuned"))
    stability_path = tables_dir / "shap_stability.csv"
    stability_df.to_csv(stability_path, index=False)
    logger.info("Saved SHAP stability table to %s", stability_path)

    logger.info("P5.T3 Explainability completed successfully.")


if __name__ == "__main__":
    run_explain()
