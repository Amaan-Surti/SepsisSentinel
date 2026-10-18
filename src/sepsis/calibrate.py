"""Calibration and Alert Policy Optimization for SepsisSentinel — P5.T1 & P5.T2.

Fits an IsotonicRegression calibrator on dev out-of-fold (OOF) probabilities,
optimizes the alerting policy (threshold, k-consecutive, sticky hours) on
calibrated probabilities to maximize normalized utility, and exports threshold curves.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from sepsis.alerting import search_policy
from sepsis.config import load_config
from sepsis.metrics import _ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fit_calibrator(
    oof_df: pd.DataFrame,
    prob_col: str = "prob",
    label_col: str = "SepsisLabel",
) -> Tuple[IsotonicRegression, Dict[str, float]]:
    """Fit IsotonicRegression on dev OOF predictions and evaluate metrics before/after."""
    y_true = oof_df[label_col].values.astype(int)
    y_prob_raw = oof_df[prob_col].values.astype(float)

    brier_before = float(brier_score_loss(y_true, y_prob_raw))
    ece_before = float(_ece(y_true, y_prob_raw, n_bins=10))

    logger.info("Before calibration: Brier = %.4f, ECE = %.4f", brier_before, ece_before)

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(y_prob_raw, y_true)

    y_prob_cal = calibrator.predict(y_prob_raw)
    brier_after = float(brier_score_loss(y_true, y_prob_cal))
    ece_after = float(_ece(y_true, y_prob_cal, n_bins=10))

    logger.info("After calibration:  Brier = %.4f, ECE = %.4f", brier_after, ece_after)

    metrics = {
        "brier_before": brier_before,
        "ece_before": ece_before,
        "brier_after": brier_after,
        "ece_after": ece_after,
    }
    return calibrator, metrics


def plot_calibration(
    y_true: np.ndarray,
    prob_raw: np.ndarray,
    prob_cal: np.ndarray,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    """Plot reliability diagram (before vs after calibration) with bin count histogram."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Diagonal reference
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    # Raw calibration curve
    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_true, prob_raw, n_bins=10, strategy="uniform"
    )
    label_raw = (
        f"Uncalibrated (Brier={metrics['brier_before']:.4f}, "
        f"ECE={metrics['ece_before']:.4f})"
    )
    ax1.plot(mean_pred_raw, frac_pos_raw, "s-", color="#D9622B", label=label_raw)

    # Calibrated curve
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, prob_cal, n_bins=10, strategy="uniform"
    )
    label_cal = (
        f"Calibrated (Brier={metrics['brier_after']:.4f}, "
        f"ECE={metrics['ece_after']:.4f})"
    )
    ax1.plot(mean_pred_cal, frac_pos_cal, "o-", color="#0E7C7B", label=label_cal)

    ax1.set_ylabel("Empirical Sepsis Rate", fontsize=11)
    ax1.set_title(
        "Dev OOF Calibration: Reliability Diagram (10 bins)", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Histogram of predicted probabilities
    bins = np.linspace(0, 1, 21)
    ax2.hist(prob_raw, bins=bins, color="#D9622B", alpha=0.5, label="Raw counts", log=True)
    ax2.hist(prob_cal, bins=bins, color="#0E7C7B", alpha=0.5, label="Calibrated counts", log=True)
    ax2.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax2.set_ylabel("Count (log)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved calibration plot to %s", out_path)


def generate_threshold_curve(
    oof_df: pd.DataFrame,
    prob_col: str,
    k: int,
    sticky: int,
    cfg: Dict[str, Any],
) -> List[Dict[str, float]]:
    """Compute performance metrics across thresholds 0.05..0.95 (step 0.01)."""
    curve = []
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    logger.info(
        "Computing threshold curve over %d points (k=%d, sticky=%d)...", len(thresholds), k, sticky
    )

    # Pre-index sorted patients
    df_sorted = oof_df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    pids = df_sorted["patient_id"].values
    labels = df_sorted["SepsisLabel"].values.astype(int)
    hours = df_sorted["hour"].values
    probs = df_sorted[prob_col].values.astype(float)
    n = len(df_sorted)

    change_mask = pids[:-1] != pids[1:]
    boundaries = np.where(change_mask)[0] + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, n]

    from sepsis.utility import compute_prediction_utility

    # Pre-classify patients
    septic_pts = []
    non_septic_pts = []
    best_total = 0.0
    inaction_total = 0.0
    total_ns_hours = 0

    for i in range(len(starts)):
        s = starts[i]
        e = ends[i]
        p_lbl = labels[s:e]
        p_hrs = hours[s:e]
        m = e - s
        if np.any(p_lbl):
            ts = float(np.argmax(p_lbl)) + 6.0
            best_preds = np.zeros(m, dtype=int)
            b_start = max(0, int(ts - 12))
            b_stop = min(int(ts + 4), m)
            if b_start < b_stop:
                best_preds[b_start:b_stop] = 1
            best_total += compute_prediction_utility(p_lbl, best_preds)
            inaction_total += compute_prediction_utility(p_lbl, np.zeros(m, dtype=int))
            septic_pts.append(
                {"start": s, "end": e, "labels": p_lbl, "hours": p_hrs, "t_sepsis": ts}
            )
        else:
            total_ns_hours += m
            non_septic_pts.append({"start": s, "end": e, "hours": m})

    denom = best_total - inaction_total

    alerts_buf = np.zeros(n, dtype=np.int8)

    for th in thresholds:
        raw = (probs >= th).astype(np.int8)

        # Apply k and sticky policy per patient
        for i in range(len(starts)):
            s = starts[i]
            e = ends[i]
            p_raw = raw[s:e]
            m = e - s
            consec = 0
            sticky_rem = 0
            for t in range(m):
                if p_raw[t]:
                    consec += 1
                else:
                    consec = 0
                if consec >= k:
                    alerts_buf[s + t] = 1
                    sticky_rem = sticky
                elif sticky_rem > 0:
                    alerts_buf[s + t] = 1
                    sticky_rem -= 1
                else:
                    alerts_buf[s + t] = 0

        # Non-septic false alerts and utility
        total_false_events = 0
        total_ns_alert_hours = 0
        for nsp in non_septic_pts:
            s = nsp["start"]
            e = nsp["end"]
            p_alrts = alerts_buf[s:e]
            total_ns_alert_hours += int(np.sum(p_alrts))
            if len(p_alrts) > 0:
                if p_alrts[0] == 1:
                    total_false_events += 1
                for idx in range(1, len(p_alrts)):
                    if p_alrts[idx] == 1 and p_alrts[idx - 1] == 0:
                        total_false_events += 1

        fp_per_day = total_false_events / (total_ns_hours / 24.0) if total_ns_hours > 0 else 0.0
        non_septic_u = -0.05 * float(total_ns_alert_hours)

        # Septic utility, sensitivity, lead time
        septic_u = 0.0
        detected = 0
        lead_times = []
        for sp in septic_pts:
            s = sp["start"]
            e = sp["end"]
            p_alrts = alerts_buf[s:e]
            p_lbl = sp["labels"]
            p_hrs = sp["hours"]
            ts = sp["t_sepsis"]

            septic_u += compute_prediction_utility(p_lbl, p_alrts)

            # sensitivity window: [ts-12, ts+3]
            win_mask = (p_hrs >= ts - 12) & (p_hrs <= ts + 3)
            if np.any(p_alrts[win_mask]):
                detected += 1

            # lead time window: [ts-48, ts+3]
            wide_mask = (p_hrs >= ts - 48) & (p_hrs <= ts + 3)
            alert_hrs_in_range = p_hrs[wide_mask & (p_alrts == 1)]
            if len(alert_hrs_in_range) > 0:
                lead_times.append(ts - alert_hrs_in_range[0])

        u_norm = (non_septic_u + septic_u - inaction_total) / denom if denom != 0 else 0.0
        sens = detected / len(septic_pts) if septic_pts else 0.0
        med_lead = float(np.median(lead_times)) if lead_times else float("nan")

        curve.append(
            {
                "threshold": float(th),
                "utility_norm": float(u_norm),
                "patient_sensitivity": float(sens),
                "false_alerts_per_patient_day": float(fp_per_day),
                "median_lead_time_h": float(med_lead),
            }
        )

    return curve


def plot_policy_tradeoff(
    curve: List[Dict[str, float]],
    best_thresh: float,
    k: int,
    sticky: int,
    out_path: Path,
) -> None:
    """Plot trade-off between utility, patient sensitivity, and false alert rate."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_curve = pd.DataFrame(curve)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    c1 = "#0E7C7B"  # Teal
    c2 = "#10242B"  # Dark Slate
    c3 = "#D9622B"  # Alert Orange

    ax1.plot(
        df_curve["threshold"],
        df_curve["utility_norm"],
        label="Normalized Utility",
        color=c1,
        lw=2.5,
    )
    ax1.plot(
        df_curve["threshold"],
        df_curve["patient_sensitivity"],
        label="Patient Sensitivity",
        color=c2,
        lw=2,
        linestyle="--",
    )
    ax1.set_xlabel("Decision Threshold (τ)", fontsize=11)
    ax1.set_ylabel("Utility / Sensitivity", fontsize=11)
    ax1.set_ylim(-0.1, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary axis for false alarms per day
    ax2 = ax1.twinx()
    ax2.plot(
        df_curve["threshold"],
        df_curve["false_alerts_per_patient_day"],
        label="False Alerts / Patient-Day",
        color=c3,
        lw=2,
        linestyle="-.",
    )
    ax2.set_ylabel("False Alerts / Pt-Day", color=c3, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=c3)

    # Highlight chosen threshold
    ax1.axvline(
        best_thresh, color="red", linestyle=":", lw=2, label=f"Selected τ = {best_thresh:.2f}"
    )

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True)

    plt.title(
        f"Alert Policy Trade-off on Calibrated Dev OOF (k={k}, sticky={sticky})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved policy tradeoff plot to %s", out_path)


def run_calibration_and_policy() -> None:
    """Orchestrate Phase 5 calibration and policy search."""
    cfg = load_config()

    oof_path = Path("models/lgbm_tuned/oof.parquet")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing {oof_path} — please run tuned training first.")

    oof_df = pd.read_parquet(oof_path)
    logger.info("Loaded OOF predictions: %d rows", len(oof_df))

    # 1. Fit calibrator
    calibrator, metrics = fit_calibrator(oof_df, prob_col="prob", label_col="SepsisLabel")

    # Save calibrator
    final_dir = Path("models/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    cal_path = final_dir / "calibrator.joblib"
    joblib.dump(calibrator, cal_path)
    logger.info("Saved calibrator to %s", cal_path)

    # Apply calibration
    prob_cal = calibrator.predict(oof_df["prob"].values.astype(float))
    oof_df["prob_calibrated"] = prob_cal

    # Save calibrated OOF
    cal_oof_path = Path("models/lgbm_tuned/oof_calibrated.parquet")
    oof_df.to_parquet(cal_oof_path, index=False)
    logger.info("Saved calibrated OOF to %s", cal_oof_path)

    # Plot reliability diagram
    fig_cal_path = Path("reports/figures/fig_calibration_dev.png")
    plot_calibration(
        y_true=oof_df["SepsisLabel"].values.astype(int),
        prob_raw=oof_df["prob"].values.astype(float),
        prob_cal=prob_cal,
        metrics=metrics,
        out_path=fig_cal_path,
    )

    # 2. Search policy on calibrated probabilities
    logger.info("Searching optimal alert policy on calibrated OOF probabilities...")
    best_params, grid_df = search_policy(oof_df, prob_col="prob_calibrated", cfg=cfg)

    # Save policy grid
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    grid_csv_path = tables_dir / "policy_grid.csv"
    grid_df.to_csv(grid_csv_path, index=False)
    logger.info("Saved policy grid to %s", grid_csv_path)

    # Save policy.json
    policy_path = final_dir / "policy.json"
    with open(policy_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Saved final policy to %s: %s", policy_path, best_params)

    # 3. Generate threshold curve and tradeoff plot
    k = best_params["k"]
    sticky = best_params["sticky"]
    tau = best_params["threshold"]

    curve = generate_threshold_curve(
        oof_df, prob_col="prob_calibrated", k=k, sticky=sticky, cfg=cfg
    )

    # Save threshold_curve.json
    curve_path = final_dir / "threshold_curve.json"
    with open(curve_path, "w") as f:
        json.dump(curve, f, indent=2)
    logger.info("Saved threshold curve to %s", curve_path)

    # Plot tradeoff
    fig_tradeoff_path = Path("reports/figures/fig_policy_tradeoff.png")
    plot_policy_tradeoff(
        curve=curve,
        best_thresh=tau,
        k=k,
        sticky=sticky,
        out_path=fig_tradeoff_path,
    )

    logger.info("P5.T1 & P5.T2 completed successfully.")


if __name__ == "__main__":
    run_calibration_and_policy()
