"""Evaluation metrics suite for SepsisSentinel — P2.T3."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from sepsis.config import load_config
from sepsis.utility import normalized_utility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helper: Expected-Calibration Error (ECE) with equal-width bins
# ---------------------------------------------------------------------------


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    total = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece_val += (count / total) * abs(avg_acc - avg_conf)
    return float(ece_val)


# ---------------------------------------------------------------------------
#  Helper: t_sepsis for a patient (first label=1 index + 6)
# ---------------------------------------------------------------------------


def _t_sepsis(labels: np.ndarray) -> float:
    """Return t_sepsis = first label=1 index + 6, or inf for non-septic."""
    if np.any(labels):
        return float(np.argmax(labels)) + 6.0
    return float("inf")


# ---------------------------------------------------------------------------
#  Helper: count alert events (rising edges)
# ---------------------------------------------------------------------------


def _count_alert_events(alerts: np.ndarray) -> int:
    """Count the number of alert events (rising edges + leading 1)."""
    if len(alerts) == 0:
        return 0
    count = int(alerts[0] == 1)
    for i in range(1, len(alerts)):
        if alerts[i] == 1 and alerts[i - 1] == 0:
            count += 1
    return count


def _alert_event_starts(alerts: np.ndarray) -> List[int]:
    """Return indices where alert events start (rising edges + leading 1)."""
    starts = []
    if len(alerts) == 0:
        return starts
    if alerts[0] == 1:
        starts.append(0)
    for i in range(1, len(alerts)):
        if alerts[i] == 1 and alerts[i - 1] == 0:
            starts.append(i)
    return starts


# ---------------------------------------------------------------------------
#  Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_predictions(
    df: pd.DataFrame,
    prob_col: str,
    alert_col: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute all Phase 2 metrics.

    Args:
        df: Must have columns patient_id, hour, SepsisLabel, <prob_col>, <alert_col>.
        prob_col: Column name for predicted probabilities.
        alert_col: Column name for binary alert decisions.
        cfg: Config dict (loaded from YAML if None).

    Returns:
        Dict of metric_name → value.
    """
    if cfg is None:
        cfg = load_config()

    # --- 1. Normalized utility ---
    labels_by_patient: Dict[str, np.ndarray] = {}
    preds_by_patient: Dict[str, np.ndarray] = {}
    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("hour")
        labels_by_patient[pid] = group["SepsisLabel"].values.astype(int)
        preds_by_patient[pid] = group[alert_col].values.astype(int)
    utility_norm = normalized_utility(labels_by_patient, preds_by_patient, cfg)

    # --- 2. Row-level metrics ---
    y_true = df["SepsisLabel"].values.astype(int)
    y_prob = df[prob_col].values.astype(float)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = float("nan")

    brier = float(brier_score_loss(y_true, y_prob))
    ece = _ece(y_true, y_prob, n_bins=10)

    # --- 3. Patient-level clinical metrics ---
    septic_patients = []
    nonseptic_patients = []
    all_alerts_col = df[alert_col].values.astype(int)

    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("hour")
        labels = group["SepsisLabel"].values.astype(int)
        alerts = group[alert_col].values.astype(int)
        hours = group["hour"].values

        if np.any(labels):
            ts = _t_sepsis(labels)
            septic_patients.append(
                {"pid": pid, "labels": labels, "alerts": alerts, "hours": hours, "t_sepsis": ts}
            )
        else:
            nonseptic_patients.append(
                {"pid": pid, "labels": labels, "alerts": alerts, "hours": hours}
            )

    # patient_sensitivity: share of septic patients with >=1 alert in [t_sepsis-12, t_sepsis+3]
    detected = 0
    lead_times = []
    for sp in septic_patients:
        ts = sp["t_sepsis"]
        alerts = sp["alerts"]
        hours = sp["hours"]
        window_mask = (hours >= ts - 12) & (hours <= ts + 3)
        if np.any(alerts[window_mask]):
            detected += 1
        # median_lead_time_h: first alert hour in [t_sepsis-48, t_sepsis+3]
        wide_mask = (hours >= ts - 48) & (hours <= ts + 3)
        alert_hours_in_range = hours[wide_mask & (alerts == 1)]
        if len(alert_hours_in_range) > 0:
            first_alert_hour = alert_hours_in_range[0]
            lead_times.append(ts - first_alert_hour)

    patient_sensitivity = detected / len(septic_patients) if septic_patients else 0.0
    median_lead_time_h = float(np.median(lead_times)) if lead_times else float("nan")

    # false_alerts_per_patient_day: alert events in non-septic patients /
    # (non-septic patient-hours / 24)
    total_false_events = 0
    total_nonseptic_hours = 0
    for nsp in nonseptic_patients:
        total_false_events += _count_alert_events(nsp["alerts"])
        total_nonseptic_hours += len(nsp["alerts"])
    false_alerts_per_patient_day = (
        total_false_events / (total_nonseptic_hours / 24.0) if total_nonseptic_hours > 0 else 0.0
    )

    # event_ppv: share of all alert events starting inside
    # [t_sepsis-12, t_sepsis+3] of a septic patient
    total_alert_events = 0
    true_alert_events = 0
    for sp in septic_patients:
        ts = sp["t_sepsis"]
        starts = _alert_event_starts(sp["alerts"])
        for s_idx in starts:
            total_alert_events += 1
            h = sp["hours"][s_idx]
            if ts - 12 <= h <= ts + 3:
                true_alert_events += 1
    for nsp in nonseptic_patients:
        total_alert_events += _count_alert_events(nsp["alerts"])
    event_ppv = true_alert_events / total_alert_events if total_alert_events > 0 else 0.0

    # alert_hour_rate: share of hours with alert = 1
    alert_hour_rate = (
        float(all_alerts_col.sum()) / len(all_alerts_col) if len(all_alerts_col) > 0 else 0.0
    )

    return {
        "utility_norm": utility_norm,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "patient_sensitivity": patient_sensitivity,
        "median_lead_time_h": median_lead_time_h,
        "false_alerts_per_patient_day": false_alerts_per_patient_day,
        "event_ppv": event_ppv,
        "alert_hour_rate": alert_hour_rate,
    }


# ---------------------------------------------------------------------------
#  Subgroup metrics
# ---------------------------------------------------------------------------


def subgroup_metrics(
    df: pd.DataFrame,
    prob_col: str,
    alert_col: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute metrics by hospital, gender, age band, and unit."""
    if cfg is None:
        cfg = load_config()

    results = []

    # Prepare age bands
    age_col = None
    for c in ["static__age", "Age"]:
        if c in df.columns:
            age_col = c
            break

    gender_col = None
    for c in ["static__gender", "Gender"]:
        if c in df.columns:
            gender_col = c
            break

    unit_col = None
    for c in ["Unit1", "Unit2", "unit"]:
        if c in df.columns:
            unit_col = c
            break

    # By hospital
    if "hospital" in df.columns:
        for hosp, sub in df.groupby("hospital"):
            try:
                m = evaluate_predictions(sub, prob_col, alert_col, cfg)
                m["subgroup_type"] = "hospital"
                m["subgroup_value"] = hosp
                results.append(m)
            except Exception:
                pass

    # By gender
    if gender_col:
        for g, sub in df.groupby(gender_col):
            try:
                m = evaluate_predictions(sub, prob_col, alert_col, cfg)
                m["subgroup_type"] = "gender"
                m["subgroup_value"] = str(g)
                results.append(m)
            except Exception:
                pass

    # By age band
    if age_col:
        bins = [0, 45, 65, 80, 200]
        labels = ["<45", "45-64", "65-79", ">=80"]
        df = df.copy()
        df["age_band"] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
        for band, sub in df.groupby("age_band", observed=True):
            if len(sub) == 0:
                continue
            try:
                m = evaluate_predictions(sub, prob_col, alert_col, cfg)
                m["subgroup_type"] = "age_band"
                m["subgroup_value"] = str(band)
                results.append(m)
            except Exception:
                pass

    # By unit (encoded as 0/1/2)
    if unit_col:
        for u_val, sub in df.groupby(unit_col):
            try:
                m = evaluate_predictions(sub, prob_col, alert_col, cfg)
                m["subgroup_type"] = "unit"
                m["subgroup_value"] = str(int(u_val))
                results.append(m)
            except Exception:
                pass

    return pd.DataFrame(results)
