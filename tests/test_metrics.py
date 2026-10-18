"""Tests for sepsis.metrics — hand-computed 3-patient synthetic example per §P2.T3."""

import numpy as np
import pandas as pd
import pytest

from sepsis.metrics import evaluate_predictions, _count_alert_events, _alert_event_starts


# Hand-crafted 3-patient example:
#
# Patient A (septic): 20 hours, labels=[0]*14 + [1]*6, t_sepsis = 14+6 = 20
#   alerts =   [0]*10 + [1,1,0,0, 1,1,1,1,1,1]
#   hours  =   [0,1,2,...,19]
#   probs  =   [0.1]*10 + [0.6,0.7,0.2,0.3, 0.8,0.9,0.85,0.9,0.95,0.95]
#
#   Alert events: event1 starts at h=10, event2 starts at h=14
#   t_sepsis = 20 → window [t_sep-12, t_sep+3] = [8, 23]
#   Both events start in [8,23] → both are true alerts
#   First alert in [t_sep-48, t_sep+3] = [-28, 23] → h=10
#   Lead time = 20 - 10 = 10 hours
#   Patient detected? Yes (alerts in [8,23])
#
# Patient B (septic): 15 hours, labels=[0]*9 + [1]*6, t_sepsis = 9+6 = 15
#   alerts = [0]*15 (no alerts at all)
#   probs  = [0.1]*15
#
#   No alert events. Patient not detected.
#   No lead time contribution.
#
# Patient C (non-septic): 10 hours, all labels=0
#   alerts = [1,1,0,0,1,0,0,1,1,0]
#   probs  = [0.6,0.7,0.1,0.2,0.5,0.1,0.1,0.8,0.9,0.1]
#
#   Alert events: event1 starts h=0, event2 starts h=4, event3 starts h=7
#   All 3 are false alert events (non-septic patient)
#   Non-septic hours = 10, false_alerts_per_patient_day = 3 / (10/24) = 7.2

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


@pytest.fixture
def synthetic_df():
    """Build the 3-patient synthetic dataframe."""
    rows = []
    # Patient A: septic, 20 hours
    labels_a = [0]*14 + [1]*6
    alerts_a = [0]*10 + [1,1,0,0, 1,1,1,1,1,1]
    probs_a  = [0.1]*10 + [0.6,0.7,0.2,0.3, 0.8,0.9,0.85,0.9,0.95,0.95]
    for h in range(20):
        rows.append({
            "patient_id": "A", "hour": h, "SepsisLabel": labels_a[h],
            "prob": probs_a[h], "alert": alerts_a[h],
        })

    # Patient B: septic, 15 hours, no alerts
    labels_b = [0]*9 + [1]*6
    for h in range(15):
        rows.append({
            "patient_id": "B", "hour": h, "SepsisLabel": labels_b[h],
            "prob": 0.1, "alert": 0,
        })

    # Patient C: non-septic, 10 hours
    alerts_c = [1,1,0,0,1,0,0,1,1,0]
    probs_c  = [0.6,0.7,0.1,0.2,0.5,0.1,0.1,0.8,0.9,0.1]
    for h in range(10):
        rows.append({
            "patient_id": "C", "hour": h, "SepsisLabel": 0,
            "prob": probs_c[h], "alert": alerts_c[h],
        })

    return pd.DataFrame(rows)


def test_alert_event_counting():
    """Test alert event counting helper."""
    alerts = np.array([1,1,0,0,1,0,0,1,1,0])
    assert _count_alert_events(alerts) == 3
    starts = _alert_event_starts(alerts)
    assert starts == [0, 4, 7]


def test_patient_sensitivity(synthetic_df):
    """Patient sensitivity: 1 of 2 septic patients detected → 0.5."""
    metrics = evaluate_predictions(synthetic_df, "prob", "alert", CFG)
    assert np.isclose(metrics["patient_sensitivity"], 0.5), (
        f"Expected 0.5, got {metrics['patient_sensitivity']}"
    )


def test_lead_time(synthetic_df):
    """Median lead time: only patient A contributes, lead=20-10=10.0 hours."""
    metrics = evaluate_predictions(synthetic_df, "prob", "alert", CFG)
    assert np.isclose(metrics["median_lead_time_h"], 10.0), (
        f"Expected 10.0, got {metrics['median_lead_time_h']}"
    )


def test_false_alerts_per_patient_day(synthetic_df):
    """3 false alert events in 10 non-septic hours → 3/(10/24) = 7.2."""
    metrics = evaluate_predictions(synthetic_df, "prob", "alert", CFG)
    expected = 3.0 / (10.0 / 24.0)
    assert np.isclose(metrics["false_alerts_per_patient_day"], expected, atol=0.01), (
        f"Expected {expected:.2f}, got {metrics['false_alerts_per_patient_day']}"
    )


def test_event_ppv(synthetic_df):
    """Event PPV: 2 true events out of 5 total = 0.4."""
    # Patient A: 2 events (both in window [8, 23]) → 2 true
    # Patient C: 3 events → 3 false
    # Total: 5 events, 2 true → PPV = 0.4
    metrics = evaluate_predictions(synthetic_df, "prob", "alert", CFG)
    assert np.isclose(metrics["event_ppv"], 0.4, atol=0.01), (
        f"Expected 0.4, got {metrics['event_ppv']}"
    )


def test_alert_hour_rate(synthetic_df):
    """Alert hour rate: (2+0+6+0+5) / 45 total hours."""
    # A: 8 alert hours, B: 0, C: 5 → 13/45
    metrics = evaluate_predictions(synthetic_df, "prob", "alert", CFG)
    expected = 13.0 / 45.0
    assert np.isclose(metrics["alert_hour_rate"], expected, atol=0.01), (
        f"Expected {expected:.4f}, got {metrics['alert_hour_rate']}"
    )


def test_metrics_are_finite(synthetic_df):
    """All metrics should be finite numbers."""
    metrics = evaluate_predictions(synthetic_df, "prob", "alert", CFG)
    for key, val in metrics.items():
        if key == "median_lead_time_h":
            continue  # can be nan if no detected patients
        assert np.isfinite(val), f"Metric {key} is not finite: {val}"
