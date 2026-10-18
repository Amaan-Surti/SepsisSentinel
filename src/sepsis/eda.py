"""Exploratory Data Analysis (EDA) module for SepsisSentinel."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sepsis.config import load_config, set_seeds
from sepsis.constants import LABS, MEASURED, SEED

logger = logging.getLogger(__name__)


def generate_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table per hospital and total."""
    # Compute patient-level stay lengths, age, gender, and sepsis status
    patient_summary = (
        df.groupby("patient_id")
        .agg(
            hospital=("hospital", "first"),
            stay_length=("hour", lambda h: h.max() + 1),
            age=("Age", "first"),
            gender=("Gender", "first"),
            is_septic=("SepsisLabel", lambda s: int((s == 1.0).any())),
        )
        .reset_index()
    )

    records = []
    groups = [
        ("A", patient_summary[patient_summary["hospital"] == "A"], df[df["hospital"] == "A"]),
        ("B", patient_summary[patient_summary["hospital"] == "B"], df[df["hospital"] == "B"]),
        ("Total", patient_summary, df),
    ]

    for name, p_sub, df_sub in groups:
        n_patients = len(p_sub)
        n_rows = len(df_sub)
        n_septic = int(p_sub["is_septic"].sum())
        pct_septic = (n_septic / n_patients) * 100.0 if n_patients > 0 else 0.0

        n_pos_rows = int((df_sub["SepsisLabel"] == 1.0).sum())
        pct_pos_rows = (n_pos_rows / n_rows) * 100.0 if n_rows > 0 else 0.0

        med_stay = float(p_sub["stay_length"].median())
        q25_stay = float(p_sub["stay_length"].quantile(0.25))
        q75_stay = float(p_sub["stay_length"].quantile(0.75))
        iqr_stay = q75_stay - q25_stay

        med_age = float(p_sub["age"].dropna().median())
        n_male = int((p_sub["gender"] == 1.0).sum())
        pct_male = (n_male / n_patients) * 100.0 if n_patients > 0 else 0.0

        records.append(
            {
                "hospital": name,
                "patients": n_patients,
                "rows": n_rows,
                "septic_patients": n_septic,
                "pct_septic_patients": pct_septic,
                "positive_rows": n_pos_rows,
                "pct_positive_rows": pct_pos_rows,
                "median_stay_length_h": med_stay,
                "iqr_stay_length_h": iqr_stay,
                "stay_length_q25_q75": f"{q25_stay:.1f}–{q75_stay:.1f}",
                "median_age": med_age,
                "pct_male": pct_male,
            }
        )

    summary_df = pd.DataFrame(records)
    return summary_df


def generate_missingness_by_var(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage of missing values overall and by hospital."""
    records = []
    df_a = df[df["hospital"] == "A"]
    df_b = df[df["hospital"] == "B"]

    for var in MEASURED:
        pct_overall = float(df[var].isna().mean() * 100.0)
        pct_a = float(df_a[var].isna().mean() * 100.0)
        pct_b = float(df_b[var].isna().mean() * 100.0)
        records.append(
            {
                "variable": var,
                "pct_missing_overall": pct_overall,
                "pct_missing_A": pct_a,
                "pct_missing_B": pct_b,
            }
        )

    return pd.DataFrame(records)


def plot_missingness_heatmap(missing_df: pd.DataFrame, out_path: Path) -> None:
    """Plot missingness heatmap of variables vs hospital."""
    plt.figure(figsize=(7, 10))
    plot_data = missing_df.set_index("variable")[
        ["pct_missing_A", "pct_missing_B", "pct_missing_overall"]
    ]
    plot_data.columns = ["Hospital A", "Hospital B", "Overall"]
    plot_data = plot_data.sort_values("Overall", ascending=True)

    sns.heatmap(
        plot_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu_r",
        cbar_kws={"label": "% Missing"},
        linewidths=0.5,
    )
    plt.title("Missingness by Variable and Hospital (%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_measurement_frequency(df: pd.DataFrame, out_path: Path) -> None:
    """Plot median measurements per patient-day for each lab."""
    stay_lens = df.groupby(["hospital", "patient_id"])["hour"].max() + 1
    patient_days = stay_lens / 24.0

    # Vectorized computation of lab counts per patient
    lab_counts = df.groupby(["hospital", "patient_id"])[LABS].count()
    lab_rates = lab_counts.div(patient_days, axis=0).reset_index()

    lab_records = []
    for var in LABS:
        med_a = float(lab_rates[lab_rates["hospital"] == "A"][var].median())
        med_b = float(lab_rates[lab_rates["hospital"] == "B"][var].median())
        lab_records.append({"lab": var, "Hospital A": med_a, "Hospital B": med_b})

    freq_df = pd.DataFrame(lab_records).set_index("lab")
    freq_df = freq_df.sort_values("Hospital A", ascending=False)

    plt.figure(figsize=(12, 6))
    freq_df.plot(kind="bar", figsize=(12, 5), width=0.8, color=["#0E7C7B", "#D9622B"])
    plt.title("Median Lab Measurements per Patient-Day (Hospital A vs B)", fontweight="bold")
    plt.xlabel("Laboratory Test")
    plt.ylabel("Measurements / Day")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_stay_length(df: pd.DataFrame, out_path: Path) -> None:
    """Plot histogram of stay length by septic status."""
    patient_summary = (
        df.groupby("patient_id")
        .agg(
            stay_length=("hour", lambda h: h.max() + 1),
            is_septic=("SepsisLabel", lambda s: "Septic" if (s == 1.0).any() else "Non-Septic"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=patient_summary,
        x="stay_length",
        hue="is_septic",
        bins=50,
        stat="density",
        common_norm=False,
        palette={"Septic": "#D9622B", "Non-Septic": "#0E7C7B"},
        kde=True,
    )
    plt.title("Distribution of ICU Stay Length by Sepsis Status", fontweight="bold")
    plt.xlabel("Stay Length (Hours)")
    plt.ylabel("Density")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_onset_timing(df: pd.DataFrame, out_path: Path) -> None:
    """Plot histogram of t_sepsis onset hour for septic patients."""
    septic_rows = df[df["SepsisLabel"] == 1.0]
    # First row with label 1 + 6
    first_onset = septic_rows.groupby("patient_id")["hour"].min() + 6

    plt.figure(figsize=(9, 5))
    sns.histplot(first_onset, bins=40, color="#D9622B", kde=True)
    plt.axvline(
        first_onset.median(),
        color="black",
        linestyle="--",
        label=f"Median: {first_onset.median():.0f}h",
    )
    plt.title("Distribution of Sepsis Clinical Onset Time ($t_{sepsis}$)", fontweight="bold")
    plt.xlabel("Hours Since ICU Admission")
    plt.ylabel("Patient Count")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_trajectories_aligned(df: pd.DataFrame, out_path: Path) -> None:
    """Plot aligned trajectories of key vitals/labs around t_sepsis vs random hour."""
    vars_to_plot = ["HR", "MAP", "Resp", "Temp", "Lactate", "WBC"]

    # Forward fill LOCF within patient
    locf_cols = ["patient_id", "hour", "SepsisLabel"] + vars_to_plot
    sub_df = df[locf_cols].copy()
    sub_df[vars_to_plot] = sub_df.groupby("patient_id")[vars_to_plot].ffill()

    # Determine t_sepsis for septic patients
    patient_status = (
        sub_df.groupby("patient_id")
        .agg(
            is_septic=("SepsisLabel", lambda s: (s == 1.0).any()),
            max_hour=("hour", "max"),
        )
        .reset_index()
    )

    septic_pids = set(patient_status[patient_status["is_septic"]]["patient_id"])
    non_septic_pts = patient_status[~patient_status["is_septic"]].copy()

    # First label index + 6
    septic_first = (
        sub_df[sub_df["SepsisLabel"] == 1.0].groupby("patient_id")["hour"].min() + 6
    ).to_dict()

    # Assign alignment hour
    rng = np.random.default_rng(SEED)
    non_septic_pts["align_hour"] = [
        rng.integers(0, max(1, m + 1)) for m in non_septic_pts["max_hour"]
    ]
    non_septic_dict = dict(zip(non_septic_pts["patient_id"], non_septic_pts["align_hour"]))

    align_dict = {**septic_first, **non_septic_dict}
    sub_df["align_hour"] = sub_df["patient_id"].map(align_dict)
    sub_df["rel_hour"] = sub_df["hour"] - sub_df["align_hour"]
    sub_df["group"] = sub_df["patient_id"].apply(
        lambda p: "Septic" if p in septic_pids else "Non-Septic"
    )

    window_df = sub_df[(sub_df["rel_hour"] >= -48) & (sub_df["rel_hour"] <= 6)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, var in enumerate(vars_to_plot):
        ax = axes[idx]
        for grp, color in [("Septic", "#D9622B"), ("Non-Septic", "#0E7C7B")]:
            grp_data = window_df[window_df["group"] == grp]
            agg = (
                grp_data.groupby("rel_hour")[var]
                .agg(
                    med="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                )
                .reset_index()
            )

            ax.plot(agg["rel_hour"], agg["med"], label=grp, color=color, lw=2)
            ax.fill_between(agg["rel_hour"], agg["q25"], agg["q75"], color=color, alpha=0.18)

        ax.axvline(
            0, color="gray", linestyle="--", alpha=0.7, label="Onset (t_sepsis)" if idx == 0 else ""
        )
        ax.set_title(var, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel("LOCF Value")
        if idx >= 3:
            ax.set_xlabel("Hours Relative to Onset")
        if idx == 0:
            ax.legend(loc="upper left")

    plt.suptitle(
        "Aligned Physiological Trajectories: Septic vs Non-Septic (Median & IQR)",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_ks_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compute two-sample KS test for observed values in Hospital A vs B."""
    records = []
    df_a = df[df["hospital"] == "A"]
    df_b = df[df["hospital"] == "B"]

    for var in MEASURED:
        vals_a = df_a[var].dropna().values
        vals_b = df_b[var].dropna().values
        if len(vals_a) > 0 and len(vals_b) > 0:
            res = stats.ks_2samp(vals_a, vals_b)
            ks_stat = float(res.statistic)
            p_val = float(res.pvalue)
        else:
            ks_stat = np.nan
            p_val = np.nan
        records.append({"variable": var, "ks_stat": ks_stat, "p_value": p_val})

    ks_df = pd.DataFrame(records).sort_values("ks_stat", ascending=False)
    return ks_df


def run_adversarial_validation(df: pd.DataFrame, out_fig: Path) -> Tuple[pd.DataFrame, float]:
    """Train LightGBM to predict hospital from patient summary features."""
    set_seeds(SEED)
    logger.info("Extracting patient-level summary features for adversarial validation...")

    # Group by patient to compute mean of observed values and pct missing
    patient_hosp = df.groupby("patient_id")["hospital"].first()
    y = (patient_hosp == "B").astype(int).values

    # Means
    means = df.groupby("patient_id")[MEASURED].mean()
    means.columns = [f"{c}__mean" for c in means.columns]

    # Missingness
    missing = df[MEASURED].isna().astype(np.float32).groupby(df["patient_id"]).mean()
    missing.columns = [f"{c}__pct_missing" for c in missing.columns]

    X = pd.concat([means, missing], axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(y), dtype=np.float32)
    feature_importances = np.zeros(X.shape[1], dtype=np.float64)

    lgb_params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": SEED,
        "deterministic": True,
        "n_jobs": -1,
    }

    logger.info("Running 5-fold adversarial validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        feature_importances += model.feature_importance(importance_type="gain") / 5.0

    auroc = float(roc_auc_score(y, oof_preds))
    logger.info("Adversarial validation OOF AUROC: %.4f", auroc)

    feat_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_gain": feature_importances,
        }
    ).sort_values("importance_gain", ascending=False)

    # Plot top 20
    top20 = feat_df.head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top20, x="importance_gain", y="feature", palette="viridis")
    plt.title(
        f"Adversarial Validation: Top 20 Features Predicting Hospital (AUROC = {auroc:.3f})",
        fontweight="bold",
    )
    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()

    res_df = pd.DataFrame([{"metric": "adversarial_auroc", "value": auroc}])
    res_df = pd.concat([res_df, feat_df], ignore_index=True)
    return res_df, auroc


def plot_class_imbalance(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot comparison of positive row percentage vs septic patient percentage."""
    tot = summary_df[summary_df["hospital"] == "Total"].iloc[0]
    data = pd.DataFrame(
        {
            "Category": ["Positive Rows\n(Hour-level)", "Septic Patients\n(Patient-level)"],
            "Percentage": [tot["pct_positive_rows"], tot["pct_septic_patients"]],
        }
    )

    plt.figure(figsize=(6, 5))
    bars = plt.bar(data["Category"], data["Percentage"], color=["#0E7C7B", "#D9622B"], width=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.15,
            f"{h:.2f}%",
            ha="center",
            fontweight="bold",
        )

    plt.title("Class Imbalance: Rows vs Patients", fontweight="bold")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, max(data["Percentage"]) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_all_eda(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run all 10 EDA analyses and save figures and tables."""
    if cfg is None:
        cfg = load_config()

    in_path = Path(cfg["paths"]["clean_rows"])
    figs_dir = Path(cfg["paths"]["figures"])
    tables_dir = Path(cfg["paths"]["tables"])

    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned dataset from %s...", in_path)
    df = pd.read_parquet(in_path)

    logger.info("1. Generating dataset_summary.csv...")
    summary_df = generate_dataset_summary(df)
    summary_df.to_csv(tables_dir / "dataset_summary.csv", index=False)

    logger.info("2. Generating missingness_by_var.csv...")
    missing_df = generate_missingness_by_var(df)
    missing_df.to_csv(tables_dir / "missingness_by_var.csv", index=False)

    logger.info("3. Plotting fig_missingness_heatmap.png...")
    plot_missingness_heatmap(missing_df, figs_dir / "fig_missingness_heatmap.png")

    logger.info("4. Plotting fig_measurement_frequency.png...")
    plot_measurement_frequency(df, figs_dir / "fig_measurement_frequency.png")

    logger.info("5. Plotting fig_stay_length.png...")
    plot_stay_length(df, figs_dir / "fig_stay_length.png")

    logger.info("6. Plotting fig_onset_timing.png...")
    plot_onset_timing(df, figs_dir / "fig_onset_timing.png")

    logger.info("7. Plotting fig_trajectories_aligned.png...")
    plot_trajectories_aligned(df, figs_dir / "fig_trajectories_aligned.png")

    logger.info("8. Generating ks_tests_A_vs_B.csv...")
    ks_df = generate_ks_tests(df)
    ks_df.to_csv(tables_dir / "ks_tests_A_vs_B.csv", index=False)

    logger.info("9. Running adversarial validation...")
    adv_df, adv_auroc = run_adversarial_validation(df, figs_dir / "fig_adversarial_top20.png")
    adv_df.to_csv(tables_dir / "adversarial_validation.csv", index=False)

    logger.info("10. Plotting fig_class_imbalance.png...")
    plot_class_imbalance(summary_df, figs_dir / "fig_class_imbalance.png")

    logger.info("All 10 EDA artifacts generated successfully.")

    # Return key findings dictionary
    tot = summary_df[summary_df["hospital"] == "Total"].iloc[0]
    findings = {
        "total_patients": int(tot["patients"]),
        "total_rows": int(tot["rows"]),
        "septic_patients": int(tot["septic_patients"]),
        "pct_septic_patients": float(tot["pct_septic_patients"]),
        "pct_positive_rows": float(tot["pct_positive_rows"]),
        "median_stay_h": float(tot["median_stay_length_h"]),
        "iqr_stay_h": float(tot["iqr_stay_length_h"]),
        "median_age": float(tot["median_age"]),
        "pct_male": float(tot["pct_male"]),
        "adversarial_auroc": adv_auroc,
    }

    tot_pts = findings["total_patients"]
    tot_rows = findings["total_rows"]
    sep_pts = findings["septic_patients"]
    pct_sep = findings["pct_septic_patients"]
    pct_pos = findings["pct_positive_rows"]
    med_stay = findings["median_stay_h"]
    iqr_stay = findings["iqr_stay_h"]
    med_age = findings["median_age"]
    pct_male = findings["pct_male"]
    adv_auc = findings["adversarial_auroc"]

    print("\n=== EDA Findings Summary ===")
    print(f"Cohort: {tot_pts:,} patients across Hospitals A and B with {tot_rows:,} hourly rows.")
    print(f"Prevalence: {sep_pts:,} septic ({pct_sep:.2f}%), {pct_pos:.2f}% positive rows.")
    print(f"Length of Stay: Median stay is {med_stay:.0f} hours (IQR {iqr_stay:.0f}h).")
    print(f"Demographics: Median age is {med_age:.1f} years, {pct_male:.1f}% male.")
    print(f"Hospital Domain Shift: Adversarial validation AUROC = {adv_auc:.4f}.")
    print("============================\n")

    return findings


def main() -> None:
    """CLI entrypoint for sepsis.eda."""
    parser = argparse.ArgumentParser(description="Generate full EDA tables and figures")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    cfg = load_config(args.config)

    run_all_eda(cfg)


if __name__ == "__main__":
    main()
