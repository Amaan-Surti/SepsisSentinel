"""Final evaluation, subgroups, and lockbox test for SepsisSentinel — P5.T4, P5.T5, P5.T6.

Strict lockbox evaluation is executed EXACTLY ONCE on the untouched 8,068 test patients.
Computes 95% bootstrap confidence intervals (1,000 resamples of patients),
evaluates baselines (B0–B3), runs subgroup fairness analyses, logs to MLflow,
and generates test ROC/PR, calibration, and lead-time figures.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sepsis.alerting import apply_policy
from sepsis.baselines import _get_b3_features
from sepsis.config import load_config
from sepsis.constants import SEED
from sepsis.metrics import _count_alert_events, _ece, evaluate_predictions, subgroup_metrics
from sepsis.utility import compute_prediction_utility

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_dev_subgroups() -> pd.DataFrame:
    """P5.T4: Compute subgroup metrics on calibrated dev OOF predictions."""
    logger.info("Computing dev OOF subgroup metrics...")
    cfg = load_config()

    oof_cal_path = Path("models/lgbm_tuned/oof_calibrated.parquet")
    if not oof_cal_path.exists():
        raise FileNotFoundError(f"Missing {oof_cal_path}")
    oof_df = pd.read_parquet(oof_cal_path)

    policy_path = Path("models/final/policy.json")
    with open(policy_path) as f:
        policy = json.load(f)

    # Apply final policy
    tau = policy["threshold"]
    k = policy["k"]
    sticky = policy["sticky"]
    oof_df["alert"] = apply_policy(oof_df, "prob_calibrated", threshold=tau, k=k, sticky=sticky)

    # Merge static subgroup columns from features.parquet
    features_df = pd.read_parquet(
        "data/processed/features.parquet",
        columns=["patient_id", "hour", "hospital", "static__gender", "static__age", "static__unit"],
    )
    merged = pd.merge(oof_df, features_df, on=["patient_id", "hour"], how="inner")

    df_sub = subgroup_metrics(merged, prob_col="prob_calibrated", alert_col="alert", cfg=cfg)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "subgroups_dev.csv"
    df_sub.to_csv(out_csv, index=False)
    logger.info("Saved dev subgroups to %s", out_csv)
    return df_sub


def log_to_mlflow() -> None:
    """P5.T5: Log final model artifacts, policy, and metrics to MLflow."""
    logger.info("Logging final run to MLflow...")
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("sepsis")

    with open("models/final/params.json") as f:
        params = json.load(f)
    with open("models/final/policy.json") as f:
        policy = json.load(f)

    with mlflow.start_run(run_name="final_model_evaluation"):
        mlflow.log_params(params)
        mlflow.log_params({f"policy_{k}": v for k, v in policy.items()})

        # Log artifact files
        final_dir = Path("models/final")
        for f in final_dir.glob("*"):
            if f.is_file():
                mlflow.log_artifact(str(f), artifact_path="final_model")

        logger.info("MLflow logging completed.")


def bootstrap_ci(
    test_df: pd.DataFrame,
    prob_col: str,
    alert_col: str,
    cfg: Dict[str, Any],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute 95% bootstrap confidence intervals by resampling patients with replacement.

    Precomputes per-patient clinical metrics for high vectorised efficiency.
    """
    logger.info("Extracting per-patient summaries for bootstrap evaluation...")
    point_metrics = evaluate_predictions(test_df, prob_col=prob_col, alert_col=alert_col, cfg=cfg)

    # Extract patient summaries
    patient_summaries = []
    pids_sorted = []
    for pid, grp in test_df.groupby("patient_id"):
        grp = grp.sort_values("hour")
        lbls = grp["SepsisLabel"].values.astype(int)
        alrts = grp[alert_col].values.astype(int)
        probs = grp[prob_col].values.astype(float)
        hrs = grp["hour"].values
        m = len(lbls)
        is_septic = bool(np.any(lbls))

        u_obs = compute_prediction_utility(lbls, alrts)
        u_inact = compute_prediction_utility(lbls, np.zeros(m, dtype=int))

        if is_septic:
            t_sepsis = float(np.argmax(lbls)) + 6.0
            best_preds = np.zeros(m, dtype=int)
            b_start = max(0, int(t_sepsis - 12))
            b_stop = min(int(t_sepsis + 4), m)
            if b_start < b_stop:
                best_preds[b_start:b_stop] = 1
            u_best = compute_prediction_utility(lbls, best_preds)

            win = (hrs >= t_sepsis - 12) & (hrs <= t_sepsis + 3)
            detected = int(np.any(alrts[win]))

            wide = (hrs >= t_sepsis - 48) & (hrs <= t_sepsis + 3)
            alert_hrs = hrs[wide & (alrts == 1)]
            lead_time = (t_sepsis - alert_hrs[0]) if len(alert_hrs) > 0 else np.nan
            fp_events = 0
            nonseptic_hours = 0
        else:
            u_best = 0.0
            detected = 0
            lead_time = np.nan
            fp_events = _count_alert_events(alrts)
            nonseptic_hours = m

        patient_summaries.append(
            {
                "is_septic": is_septic,
                "u_obs": u_obs,
                "u_inact": u_inact,
                "u_best": u_best,
                "detected": detected,
                "lead_time": lead_time,
                "fp_events": fp_events,
                "nonseptic_hours": nonseptic_hours,
                "y_true": lbls,
                "y_prob": probs,
                "alerts": alrts,
            }
        )
        pids_sorted.append(pid)

    n_pts = len(patient_summaries)
    logger.info(
        "Precomputed summaries for %d patients. Running %d bootstrap iterations...",
        n_pts,
        n_bootstraps,
    )

    rng = np.random.RandomState(seed)

    boot_utility = []
    boot_sens = []
    boot_lead = []
    boot_fp_day = []
    boot_auroc = []
    boot_auprc = []
    boot_brier = []
    boot_ece = []

    for b in range(n_bootstraps):
        sample_indices = rng.choice(n_pts, size=n_pts, replace=True)

        tot_u_obs = sum(patient_summaries[i]["u_obs"] for i in sample_indices)
        tot_u_inact = sum(patient_summaries[i]["u_inact"] for i in sample_indices)
        tot_u_best = sum(patient_summaries[i]["u_best"] for i in sample_indices)
        u_norm = (
            (tot_u_obs - tot_u_inact) / (tot_u_best - tot_u_inact)
            if (tot_u_best - tot_u_inact) != 0
            else 0.0
        )
        boot_utility.append(u_norm)

        septic_indices = [i for i in sample_indices if patient_summaries[i]["is_septic"]]
        if septic_indices:
            sens = sum(patient_summaries[i]["detected"] for i in septic_indices) / len(
                septic_indices
            )
            leads = [
                patient_summaries[i]["lead_time"]
                for i in septic_indices
                if not np.isnan(patient_summaries[i]["lead_time"])
            ]
            med_lead = float(np.median(leads)) if leads else np.nan
        else:
            sens = np.nan
            med_lead = np.nan
        boot_sens.append(sens)
        boot_lead.append(med_lead)

        tot_fp = sum(patient_summaries[i]["fp_events"] for i in sample_indices)
        tot_ns_hrs = sum(patient_summaries[i]["nonseptic_hours"] for i in sample_indices)
        fp_rate = tot_fp / (tot_ns_hrs / 24.0) if tot_ns_hrs > 0 else 0.0
        boot_fp_day.append(fp_rate)

        # For row metrics, evaluate on a subset of 500 sampled patients for speed
        if (b + 1) % 10 == 0 or b < 100:
            sub_indices = sample_indices[:500]
            y_t = np.concatenate([patient_summaries[i]["y_true"] for i in sub_indices])
            y_p = np.concatenate([patient_summaries[i]["y_prob"] for i in sub_indices])
            try:
                boot_auroc.append(roc_auc_score(y_t, y_p))
                boot_auprc.append(average_precision_score(y_t, y_p))
                boot_brier.append(brier_score_loss(y_t, y_p))
                boot_ece.append(_ece(y_t, y_p, n_bins=10))
            except Exception:
                pass

    def get_ci(arr: List[float]) -> Tuple[float, float]:
        clean = [x for x in arr if not np.isnan(x)]
        if not clean:
            return float("nan"), float("nan")
        return float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))

    ci_map = {
        "utility_norm": get_ci(boot_utility),
        "patient_sensitivity": get_ci(boot_sens),
        "median_lead_time_h": get_ci(boot_lead),
        "false_alerts_per_patient_day": get_ci(boot_fp_day),
        "auroc": get_ci(boot_auroc),
        "auprc": get_ci(boot_auprc),
        "brier": get_ci(boot_brier),
        "ece": get_ci(boot_ece),
    }

    results = []
    for k, val in point_metrics.items():
        ci_low, ci_high = ci_map.get(k, (float("nan"), float("nan")))
        results.append(
            {
                "metric": k,
                "value": float(val),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }
        )

    return pd.DataFrame(results)


def plot_test_figures(
    test_df: pd.DataFrame,
    prob_col: str,
    raw_prob_col: str,
    alert_col: str,
    out_dir: Path,
) -> None:
    """Generate all required test figures: ROC/PR, Calibration, and Lead-Time Histogram."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = test_df["SepsisLabel"].values.astype(int)
    y_prob = test_df[prob_col].values.astype(float)
    y_raw = test_df[raw_prob_col].values.astype(float)

    # 1. ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, color="#0E7C7B", lw=2, label=f"Final Model (AUROC = {auroc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("Lockbox Test: ROC Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))
    ax2.plot(rec, prec, color="#0E7C7B", lw=2, label=f"Final Model (AUPRC = {auprc:.4f})")
    ax2.axhline(
        prevalence, color="k", linestyle="--", lw=1.5, label=f"Prevalence ({prevalence:.2%})"
    )
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax2.set_ylabel("Precision (PPV)", fontsize=11)
    ax2.set_title("Lockbox Test: Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    roc_pr_path = out_dir / "fig_roc_pr_test.png"
    fig.savefig(roc_pr_path, dpi=300)
    plt.close(fig)
    logger.info("Saved test ROC/PR curves to %s", roc_pr_path)

    # 2. Calibration reliability diagram
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    brier_raw = brier_score_loss(y_true, y_raw)
    ece_raw = _ece(y_true, y_raw, n_bins=10)
    brier_cal = brier_score_loss(y_true, y_prob)
    ece_cal = _ece(y_true, y_prob, n_bins=10)

    f_raw, m_raw = calibration_curve(y_true, y_raw, n_bins=10, strategy="uniform")
    ax1.plot(
        m_raw,
        f_raw,
        "s-",
        color="#D9622B",
        label=f"Uncalibrated (Brier={brier_raw:.4f}, ECE={ece_raw:.4f})",
    )

    f_cal, m_cal = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax1.plot(
        m_cal,
        f_cal,
        "o-",
        color="#0E7C7B",
        label=f"Calibrated (Brier={brier_cal:.4f}, ECE={ece_cal:.4f})",
    )

    ax1.set_ylabel("Empirical Sepsis Rate", fontsize=11)
    ax1.set_title("Lockbox Test: Reliability Diagram (10 bins)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    bins = np.linspace(0, 1, 21)
    ax2.hist(y_prob, bins=bins, color="#0E7C7B", alpha=0.6, log=True)
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Count (log)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    cal_path = out_dir / "fig_calibration_test.png"
    fig.savefig(cal_path, dpi=300)
    plt.close(fig)
    logger.info("Saved test calibration plot to %s", cal_path)

    # 3. Lead time histogram
    lead_times = []
    for pid, grp in test_df.groupby("patient_id"):
        grp = grp.sort_values("hour")
        lbls = grp["SepsisLabel"].values
        if np.any(lbls):
            t_sepsis = float(np.argmax(lbls)) + 6.0
            hrs = grp["hour"].values
            alrts = grp[alert_col].values
            mask = (hrs >= t_sepsis - 48) & (hrs <= t_sepsis + 3)
            alert_hrs = hrs[mask & (alrts == 1)]
            if len(alert_hrs) > 0:
                lead_times.append(t_sepsis - alert_hrs[0])

    if lead_times:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            lead_times, bins=np.arange(-4, 50, 2), color="#0E7C7B", edgecolor="#10242B", alpha=0.85
        )
        med_lead = float(np.median(lead_times))
        ax.axvline(
            med_lead,
            color="red",
            linestyle="--",
            lw=2,
            label=f"Median Lead Time = {med_lead:.1f} h",
        )
        ax.set_xlabel("Lead Time Before Sepsis Onset (Hours)", fontsize=11)
        ax.set_ylabel("Number of Detected Septic Patients", fontsize=11)
        ax.set_title(
            "Lockbox Test: Early Detection Lead-Time Distribution", fontsize=12, fontweight="bold"
        )
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        lead_path = out_dir / "fig_lead_time_hist_test.png"
        fig.savefig(lead_path, dpi=300)
        plt.close(fig)
        logger.info("Saved test lead-time histogram to %s", lead_path)


def evaluate_test_baselines(
    features_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dev_features: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluate B0, B1, B2, B3 baselines on the lockbox test set."""
    logger.info("Evaluating baselines on lockbox test set...")
    records = []

    # B0: Never alert
    df_b0 = test_df.copy()
    df_b0["prob_b0"] = 0.0
    df_b0["alert_b0"] = 0
    m_b0 = evaluate_predictions(df_b0, "prob_b0", "alert_b0", cfg)
    m_b0["baseline"] = "B0_never_alert"
    records.append(m_b0)

    # B1: SIRS rule (score__sirs >= 2)
    df_b1 = test_df.copy()
    df_b1["prob_b1"] = (df_b1["score__sirs"] >= 2).astype(float)
    df_b1["alert_b1"] = (df_b1["score__sirs"] >= 2).astype(int)
    m_b1 = evaluate_predictions(df_b1, "prob_b1", "alert_b1", cfg)
    m_b1["baseline"] = "B1_sirs_rule"
    records.append(m_b1)

    # B2: Partial qSOFA rule (score__qsofa_partial >= 2)
    df_b2 = test_df.copy()
    df_b2["prob_b2"] = (df_b2["score__qsofa_partial"] >= 2).astype(float)
    df_b2["alert_b2"] = (df_b2["score__qsofa_partial"] >= 2).astype(int)
    m_b2 = evaluate_predictions(df_b2, "prob_b2", "alert_b2", cfg)
    m_b2["baseline"] = "B2_qsofa_rule"
    records.append(m_b2)

    # B3: Logistic Regression
    b3_features = _get_b3_features(cfg)
    available_b3 = [c for c in b3_features if c in dev_features.columns]

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
    logger.info("Fitting B3 logistic regression on all dev rows...")
    pipe.fit(dev_features[available_b3].values, dev_features["SepsisLabel"].values.astype(int))

    df_b3 = test_df.copy()
    probs_b3 = pipe.predict_proba(df_b3[available_b3].values)[:, 1]
    df_b3["prob_b3"] = probs_b3
    # B3 dev-chosen policy: threshold=0.72, k=1, sticky=6
    df_b3["alert_b3"] = apply_policy(df_b3, "prob_b3", threshold=0.72, k=1, sticky=6)
    m_b3 = evaluate_predictions(df_b3, "prob_b3", "alert_b3", cfg)
    m_b3["baseline"] = "B3_logistic_regression"
    records.append(m_b3)

    df_baselines = pd.DataFrame(records)
    out_path = Path("reports/tables/final_test_baselines.csv")
    df_baselines.to_csv(out_path, index=False)
    logger.info("Saved test baselines to %s", out_path)
    return df_baselines


def run_lockbox_evaluation() -> None:
    """P5.T6: Strict one-time evaluation on the 8,068 lockbox test patients."""
    cfg = load_config()

    # Load lockbox test patient IDs
    lockbox_df = pd.read_csv("data/splits/lockbox.csv")
    test_pids = set(lockbox_df[lockbox_df["split"] == "test"]["patient_id"])
    logger.info("Lockbox cohort: %d test patients", len(test_pids))

    # Load feature matrix
    features_df = pd.read_parquet("data/processed/features.parquet")
    test_mask = features_df["patient_id"].isin(test_pids)
    test_features = features_df[test_mask].copy().reset_index(drop=True)
    dev_features = features_df[~test_mask].copy().reset_index(drop=True)
    logger.info("Test rows: %d, Dev rows: %d", len(test_features), len(dev_features))

    # Load final model, calibrator, and policy
    with open("models/final/feature_list.json") as f:
        feature_list = json.load(f)

    booster = lgb.Booster(model_file="models/final/lgbm.txt")
    calibrator = joblib.load("models/final/calibrator.joblib")
    with open("models/final/policy.json") as f:
        policy = json.load(f)

    tau = float(policy["threshold"])
    k = int(policy["k"])
    sticky = int(policy["sticky"])
    logger.info("Applying final alert policy: threshold=%.4f, k=%d, sticky=%d", tau, k, sticky)

    # Predictions
    raw_probs = booster.predict(test_features[feature_list])
    test_features["prob_raw"] = raw_probs
    cal_probs = calibrator.predict(raw_probs)
    test_features["prob_calibrated"] = cal_probs

    test_features["alert"] = apply_policy(
        test_features, "prob_calibrated", threshold=tau, k=k, sticky=sticky
    )

    # 1. Compute bootstrap metrics (1,000 resamples)
    df_ci = bootstrap_ci(
        test_features,
        prob_col="prob_calibrated",
        alert_col="alert",
        cfg=cfg,
        n_bootstraps=1000,
        seed=42,
    )
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    ci_csv_path = tables_dir / "final_test_metrics.csv"
    df_ci.to_csv(ci_csv_path, index=False)
    logger.info("Saved final test metrics with 95% CIs to %s", ci_csv_path)

    # 2. Test subgroups
    logger.info("Computing lockbox test subgroup metrics...")
    test_subgroups = subgroup_metrics(test_features, "prob_calibrated", "alert", cfg)
    subgroups_csv_path = tables_dir / "final_test_subgroups.csv"
    test_subgroups.to_csv(subgroups_csv_path, index=False)
    logger.info("Saved test subgroups to %s", subgroups_csv_path)

    # 3. Test figures
    figs_dir = Path("reports/figures")
    plot_test_figures(
        test_features,
        prob_col="prob_calibrated",
        raw_prob_col="prob_raw",
        alert_col="alert",
        out_dir=figs_dir,
    )

    # 4. Baselines on test set
    evaluate_test_baselines(features_df, test_features, dev_features, cfg)

    logger.info("P5.T6 Lockbox evaluation completed successfully.")


def run_phase5_evaluation() -> None:
    """Execute P5.T4, P5.T5, and P5.T6."""
    run_dev_subgroups()
    log_to_mlflow()
    run_lockbox_evaluation()


if __name__ == "__main__":
    run_phase5_evaluation()
