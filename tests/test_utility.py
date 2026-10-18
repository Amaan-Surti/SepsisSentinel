"""Tests for sepsis.utility — hand-verified acceptance checks per PLAN.md §P2.T2."""

import numpy as np
import pytest

from sepsis.utility import compute_prediction_utility, normalized_utility


# Default config for utility
CFG = {
    "utility": {
        "dt_early": -12,
        "dt_optimal": -6,
        "dt_late": 3,
        "max_u_tp": 1.0,
        "min_u_fn": -2.0,
        "u_fp": -0.05,
        "u_tn": 0.0,
    }
}


def test_nonseptic_all_zero():
    """Non-septic, n=10, all-zero predictions → utility 0.0."""
    labels = np.zeros(10)
    preds = np.zeros(10)
    u = compute_prediction_utility(labels, preds)
    assert u == 0.0


def test_nonseptic_all_one():
    """Non-septic, n=10, all-one predictions → −0.5."""
    labels = np.zeros(10)
    preds = np.ones(10)
    u = compute_prediction_utility(labels, preds)
    assert np.isclose(u, -0.5), f"Expected -0.5, got {u}"


def test_septic_all_zero():
    """Septic, n=40, labels 1 from index 20, all-zero predictions → −10.0.

    t_sepsis = 20 + 6 = 26. FN penalty from t=21..29 (9 hours in optimal..late window):
    dt_optimal = -6, dt_late = 3
    For t in [t_sepsis + dt_optimal + 1, t_sepsis + dt_late] = [21, 29]:
      m_3 = min_u_fn / (dt_late - dt_optimal) = -2/9
      b_3 = -m_3 * dt_optimal = -(−2/9)*(−6) = −12/9 = −4/3
      u[t] = m_3 * (t - t_sepsis) + b_3
    Sum of penalties from t=21 to t=29: each dt = t-26, t ranges -5..3
    Actually for all-zero preds on septic patient:
    - hours 0..20: labels are 0, preds are 0 → u_tn = 0 each
    - hours 20: first label=1, u[20] = 0 (t <= t_sepsis + dt_optimal)
    - hours 21..29 (t_sepsis+dt_optimal < t <= t_sepsis+dt_late):
      FN penalties with m_3, b_3
    Total = −10.0 as stated in the plan.
    """
    labels = np.zeros(40)
    labels[20:] = 1
    preds = np.zeros(40)
    u = compute_prediction_utility(labels, preds)
    assert np.isclose(u, -10.0), f"Expected -10.0, got {u}"


def test_septic_all_one():
    """Septic, n=40, labels 1 from index 20, all-one predictions → 6.8."""
    labels = np.zeros(40)
    labels[20:] = 1
    preds = np.ones(40)
    u = compute_prediction_utility(labels, preds)
    assert np.isclose(u, 6.8, atol=1e-6), f"Expected 6.8, got {u}"


def test_septic_single_pred_at_optimal():
    """Septic, prediction=1 only at index 20 → −9.0.

    t_sepsis = 26. Prediction 1 at t=20 (which is t_sepsis + dt_optimal = 20).
    Contribution at t=20: TP with m_1*(20-26) + b_1 = (1/6)*(-6) + 2 = 1.0
    FN penalties at t=21..29 = −10.0
    Total = 1.0 + (−10.0) = −9.0
    """
    labels = np.zeros(40)
    labels[20:] = 1
    preds = np.zeros(40)
    preds[20] = 1
    u = compute_prediction_utility(labels, preds)
    assert np.isclose(u, -9.0, atol=1e-6), f"Expected -9.0, got {u}"


def test_normalized_utility_all_one_septic():
    """Normalized utility for single septic patient with all-one preds → 0.96.

    observed = 6.8, inaction = -10.0, best = 7.5
    normalized = (6.8 - (-10)) / (7.5 - (-10)) = 16.8 / 17.5 = 0.96
    """
    labels = np.zeros(40)
    labels[20:] = 1
    preds = np.ones(40)

    labels_dict = {"p1": labels}
    preds_dict = {"p1": preds}

    nu = normalized_utility(labels_dict, preds_dict, CFG)
    assert np.isclose(nu, 0.96, atol=0.01), f"Expected ~0.96, got {nu}"


def test_normalized_utility_all_zero_is_zero():
    """Normalized utility of all-zero predictions on any set = 0.0."""
    # Mix of septic and non-septic
    labels_dict = {
        "p1": np.zeros(10),  # non-septic
        "p2": np.concatenate([np.zeros(20), np.ones(20)]),  # septic
    }
    preds_dict = {
        "p1": np.zeros(10),
        "p2": np.zeros(40),
    }
    nu = normalized_utility(labels_dict, preds_dict, CFG)
    assert np.isclose(nu, 0.0, atol=1e-10), f"Expected 0.0, got {nu}"


def test_normalized_utility_best_is_one():
    """Normalized utility of best predictions = 1.0."""
    labels_septic = np.zeros(40)
    labels_septic[20:] = 1
    # t_sepsis = 26, best preds: 1 from max(0, 26-12)=14 to min(26+3+1,40)=30 exclusive
    best_preds = np.zeros(40)
    best_preds[14:30] = 1

    labels_dict = {
        "p1": np.zeros(10),  # non-septic
        "p2": labels_septic,
    }
    preds_dict = {
        "p1": np.zeros(10),
        "p2": best_preds,
    }
    nu = normalized_utility(labels_dict, preds_dict, CFG)
    assert np.isclose(nu, 1.0, atol=1e-6), f"Expected 1.0, got {nu}"
