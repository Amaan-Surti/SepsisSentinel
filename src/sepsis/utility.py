"""Official Challenge normalized utility scoring for SepsisSentinel."""

import argparse
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sepsis.config import load_config

logger = logging.getLogger(__name__)


def compute_prediction_utility(
    labels,
    predictions,
    dt_early=-12,
    dt_optimal=-6,
    dt_late=3.0,
    max_u_tp=1,
    min_u_fn=-2,
    u_fp=-0.05,
    u_tn=0,
):
    """Compute the Challenge utility score for a single patient."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if np.any(labels):
        is_septic = True
        t_sepsis = int(np.argmax(labels)) - dt_optimal  # first label index + 6
    else:
        is_septic = False
        t_sepsis = float("inf")
    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return float(np.sum(u))


def _best_predictions_for_patient(labels, dt_early, dt_optimal, dt_late):
    """Return the best possible binary predictions for a patient."""
    labels = np.asarray(labels)
    n = len(labels)
    best = np.zeros(n, dtype=int)
    if np.any(labels):
        t_sepsis = int(np.argmax(labels)) - dt_optimal
        start = max(0, int(t_sepsis + dt_early))
        end = min(int(t_sepsis + dt_late + 1), n)
        best[start:end] = 1
    return best


def normalized_utility(
    labels_by_patient: Dict[str, np.ndarray],
    preds_by_patient: Dict[str, np.ndarray],
    cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute Challenge normalized utility across all patients."""
    if cfg is None:
        cfg = load_config()
    ucfg = cfg.get("utility", {})
    dt_early = ucfg.get("dt_early", -12)
    dt_optimal = ucfg.get("dt_optimal", -6)
    dt_late = ucfg.get("dt_late", 3)
    max_u_tp = ucfg.get("max_u_tp", 1.0)
    min_u_fn = ucfg.get("min_u_fn", -2.0)
    u_fp = ucfg.get("u_fp", -0.05)
    u_tn = ucfg.get("u_tn", 0.0)

    observed = 0.0
    best_total = 0.0
    inaction = 0.0

    for pid, labels in labels_by_patient.items():
        preds = preds_by_patient[pid]
        observed += compute_prediction_utility(
            labels,
            preds,
            dt_early,
            dt_optimal,
            dt_late,
            max_u_tp,
            min_u_fn,
            u_fp,
            u_tn,
        )
        best_preds = _best_predictions_for_patient(labels, dt_early, dt_optimal, dt_late)
        best_total += compute_prediction_utility(
            labels,
            best_preds,
            dt_early,
            dt_optimal,
            dt_late,
            max_u_tp,
            min_u_fn,
            u_fp,
            u_tn,
        )
        zeros = np.zeros(len(labels), dtype=int)
        inaction += compute_prediction_utility(
            labels,
            zeros,
            dt_early,
            dt_optimal,
            dt_late,
            max_u_tp,
            min_u_fn,
            u_fp,
            u_tn,
        )

    denom = best_total - inaction
    if denom == 0:
        return 0.0
    return (observed - inaction) / denom


def patient_utilities(
    labels_by_patient: Dict[str, np.ndarray],
    preds_by_patient: Dict[str, np.ndarray],
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Return per-patient observed utility for debugging."""
    if cfg is None:
        cfg = load_config()
    ucfg = cfg.get("utility", {})
    records = []
    for pid, labels in labels_by_patient.items():
        preds = preds_by_patient[pid]
        u = compute_prediction_utility(
            labels,
            preds,
            ucfg.get("dt_early", -12),
            ucfg.get("dt_optimal", -6),
            ucfg.get("dt_late", 3),
            ucfg.get("max_u_tp", 1.0),
            ucfg.get("min_u_fn", -2.0),
            ucfg.get("u_fp", -0.05),
            ucfg.get("u_tn", 0.0),
        )
        records.append({"patient_id": pid, "utility": u})
    return pd.DataFrame(records)


def main() -> None:
    """CLI entrypoint for sepsis.utility."""
    parser = argparse.ArgumentParser(description="Utility scoring module")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.parse_args()
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger.info("Utility module loaded. Use from code: from sepsis.utility import ...")


if __name__ == "__main__":
    main()
