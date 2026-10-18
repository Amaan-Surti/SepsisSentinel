"""Alert policy for SepsisSentinel — P2.T4.

Turn probability predictions into binary alerts using threshold + k-consecutive + sticky rules.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sepsis.config import load_config
from sepsis.metrics import evaluate_predictions

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
    alerts = np.zeros(len(df), dtype=int)

    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("hour")
        idx = group.index
        probs = group[prob_col].values

        # Step 1: raw threshold
        raw = (probs >= threshold).astype(int)

        # Step 2: k-consecutive rule
        k_pass = np.zeros(len(raw), dtype=int)
        for t in range(len(raw)):
            if t < k - 1:
                # Not enough prior hours
                if k == 1 and raw[t]:
                    k_pass[t] = 1
                continue
            if np.all(raw[t - k + 1: t + 1]):
                k_pass[t] = 1

        # Step 3: sticky
        final = np.zeros(len(raw), dtype=int)
        last_fire = -999
        for t in range(len(raw)):
            if k_pass[t]:
                final[t] = 1
                last_fire = t
            elif t - last_fire <= sticky and last_fire >= 0:
                final[t] = 1

        alerts[idx] = final

    return alerts


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

    records: List[Dict[str, Any]] = []
    best_utility = -np.inf
    best_params: Dict[str, Any] = {}

    total = len(thresholds) * len(k_values) * len(sticky_values)
    logger.info("Searching %d policy configurations...", total)

    for k in k_values:
        for sticky in sticky_values:
            for thresh in thresholds:
                alert_col = "__tmp_alert__"
                df_tmp = df.copy()
                df_tmp[alert_col] = apply_policy(df_tmp, prob_col, float(thresh), k, sticky)

                try:
                    m = evaluate_predictions(df_tmp, prob_col, alert_col, cfg)
                    u = m["utility_norm"]
                except Exception:
                    u = float("-inf")

                records.append({
                    "threshold": float(thresh),
                    "k": k,
                    "sticky": sticky,
                    "utility_norm": u,
                })

                # Ties → choose the higher threshold
                if u > best_utility or (u == best_utility and float(thresh) > best_params.get("threshold", 0)):
                    best_utility = u
                    best_params = {
                        "threshold": float(thresh),
                        "k": k,
                        "sticky": sticky,
                        "utility_norm": u,
                    }

    grid_df = pd.DataFrame(records)
    logger.info("Best policy: %s", best_params)

    return best_params, grid_df
