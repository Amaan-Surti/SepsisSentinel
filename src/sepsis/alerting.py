"""Alert policy for SepsisSentinel — P2.T4.

Turn probability predictions into binary alerts using threshold + k-consecutive + sticky rules.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sepsis.config import load_config

logger = logging.getLogger(__name__)


def apply_policy(
    df: pd.DataFrame,
    prob_col: str,
    threshold: float,
    k: int,
    sticky: int,
) -> np.ndarray:
    """Apply alert policy per patient, in hour order.

    1. raw = prob >= threshold
    2. k-consecutive: alert at t only if raw is true for hours t-k+1..t (all within stay).
    3. sticky: once k-rule fires at t, keep alert=1 for the next `sticky` hours.

    Args:
        df: Must have columns patient_id, hour, <prob_col>.
        prob_col: Column name for predicted probabilities.
        threshold: Probability threshold.
        k: Number of consecutive hours required.
        sticky: Number of extra hours to keep alert after k-rule fires.

    Returns:
        np.ndarray of binary alerts aligned with df's index order.
    """
    df_sorted = df.sort_values(["patient_id", "hour"])
    orig_idx = df_sorted.index

    pids = df_sorted["patient_id"].values
    probs = df_sorted[prob_col].values
    n = len(df_sorted)

    if n == 0:
        return np.zeros(0, dtype=int)

    change_mask = pids[:-1] != pids[1:]
    boundaries = np.where(change_mask)[0] + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, n]

    raw = (probs >= threshold).astype(np.int8)
    alerts = np.zeros(n, dtype=int)

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
                alerts[s + t] = 1
                sticky_rem = sticky
            elif sticky_rem > 0:
                alerts[s + t] = 1
                sticky_rem -= 1

    result = pd.Series(alerts, index=orig_idx)
    return result.loc[df.index].values


def search_policy(
    df: pd.DataFrame,
    prob_col: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Grid search over alert policy parameters to maximize normalized utility.

    Args:
        df: Must have columns patient_id, hour, SepsisLabel, <prob_col>.
        prob_col: Column name for predicted probabilities.
        cfg: Config dict.

    Returns:
        (best_params, grid_df) where best_params is dict with threshold, k, sticky
        and grid_df is the full grid results.
    """
    from sepsis.utility import compute_prediction_utility

    if cfg is None:
        cfg = load_config()

    pcfg = cfg.get("alert_policy", {})
    thresh_start = pcfg.get("threshold_grid_start", 0.01)
    thresh_stop = pcfg.get("threshold_grid_stop", 0.99)
    thresh_step = pcfg.get("threshold_grid_step", 0.01)
    k_values = pcfg.get("k_consecutive", [1, 2, 3])
    sticky_values = pcfg.get("sticky_hours", [0, 6])

    thresholds = np.arange(thresh_start, thresh_stop + 1e-9, thresh_step)
    thresholds = np.round(thresholds, 4)

    df_sorted = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    pids = df_sorted["patient_id"].values
    probs = df_sorted[prob_col].values
    labels = df_sorted["SepsisLabel"].values.astype(int)
    n = len(df_sorted)

    change_mask = pids[:-1] != pids[1:]
    boundaries = np.where(change_mask)[0] + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, n]

    # Pre-classify patients into septic and non-septic
    dt_early = -12
    dt_optimal = -6
    dt_late = 3.0
    septic_indices = []
    septic_labels_list = []
    non_septic_slice_masks = np.zeros(n, dtype=bool)

    best_total = 0.0
    inaction_total = 0.0

    for i in range(len(starts)):
        s = starts[i]
        e = ends[i]
        p_lbl = labels[s:e]
        if np.any(p_lbl):
            septic_indices.append(i)
            septic_labels_list.append(p_lbl)
            m = len(p_lbl)
            t_sepsis = np.argmax(p_lbl) - dt_optimal
            best_preds = np.zeros(m, dtype=int)
            b_start = max(0, int(t_sepsis + dt_early))
            b_stop = min(int(t_sepsis + dt_late + 1), m)
            if b_start < b_stop:
                best_preds[b_start:b_stop] = 1
            best_total += compute_prediction_utility(p_lbl, best_preds)
            inaction_total += compute_prediction_utility(p_lbl, np.zeros(m, dtype=int))
        else:
            non_septic_slice_masks[s:e] = True

    denom = best_total - inaction_total

    records: List[Dict[str, Any]] = []
    best_utility = -np.inf
    best_params: Dict[str, Any] = {}

    total = len(thresholds) * len(k_values) * len(sticky_values)
    logger.info("Searching %d policy configurations (fast vectorised search)...", total)

    alerts_buf = np.zeros(n, dtype=np.int8)

    for k in k_values:
        for sticky in sticky_values:
            for thresh in thresholds:
                raw = (probs >= thresh).astype(np.int8)

                # Compute alerts per patient
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

                # Non-septic utility: -0.05 per alert hour
                non_septic_u = -0.05 * float(np.sum(alerts_buf[non_septic_slice_masks]))

                # Septic utility
                septic_u = 0.0
                for idx_in_septic, pt_idx in enumerate(septic_indices):
                    s = starts[pt_idx]
                    e = ends[pt_idx]
                    p_alerts = alerts_buf[s:e]
                    p_lbl = septic_labels_list[idx_in_septic]
                    septic_u += compute_prediction_utility(p_lbl, p_alerts)

                obs_total = non_septic_u + septic_u
                u = (obs_total - inaction_total) / denom if denom != 0 else 0.0

                records.append(
                    {
                        "threshold": float(thresh),
                        "k": k,
                        "sticky": sticky,
                        "utility_norm": float(u),
                    }
                )

                # Ties → choose the higher threshold
                if u > best_utility or (
                    u == best_utility and float(thresh) > best_params.get("threshold", 0)
                ):
                    best_utility = u
                    best_params = {
                        "threshold": float(thresh),
                        "k": k,
                        "sticky": sticky,
                        "utility_norm": float(u),
                    }

    grid_df = pd.DataFrame(records)
    logger.info("Best policy: %s", best_params)

    return best_params, grid_df
