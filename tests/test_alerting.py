"""Tests for sepsis.alerting — acceptance checks per §P2.T4."""

import numpy as np
import pandas as pd
import pytest

from sepsis.alerting import apply_policy


def _make_df(probs: list, patient_id: str = "P1") -> pd.DataFrame:
    """Helper: make a single-patient df from a list of probabilities."""
    return pd.DataFrame({
        "patient_id": [patient_id] * len(probs),
        "hour": list(range(len(probs))),
        "prob": probs,
    })


class TestKConsecutive:
    """Test that k-consecutive rule works correctly."""

    def test_k1_any_above_threshold(self):
        """k=1: alert whenever prob >= threshold."""
        df = _make_df([0.1, 0.5, 0.3, 0.8, 0.2])
        alerts = apply_policy(df, "prob", threshold=0.5, k=1, sticky=0)
        expected = [0, 1, 0, 1, 0]
        np.testing.assert_array_equal(alerts, expected)

    def test_k2_requires_two_consecutive(self):
        """k=2: alert only when 2 consecutive hours above threshold."""
        df = _make_df([0.1, 0.6, 0.3, 0.7, 0.8, 0.2])
        alerts = apply_policy(df, "prob", threshold=0.5, k=2, sticky=0)
        # h0: below → 0
        # h1: above, but h0 below → 0
        # h2: below → 0
        # h3: above, but h2 below → 0
        # h4: above, and h3 above → 1
        # h5: below → 0
        expected = [0, 0, 0, 0, 1, 0]
        np.testing.assert_array_equal(alerts, expected)

    def test_k3_requires_three_consecutive(self):
        """k=3: alert only when 3 consecutive hours above threshold."""
        df = _make_df([0.6, 0.7, 0.8, 0.3, 0.6, 0.7, 0.8, 0.9])
        alerts = apply_policy(df, "prob", threshold=0.5, k=3, sticky=0)
        # h0,h1: not enough history → 0
        # h2: h0,h1,h2 all above → 1
        # h3: below → 0
        # h4,h5: not 3 consecutive above yet → 0 for h4, 0 for h5 (h3 below)
        # h6: h4,h5,h6 all above → 1
        # h7: h5,h6,h7 all above → 1
        expected = [0, 0, 1, 0, 0, 0, 1, 1]
        np.testing.assert_array_equal(alerts, expected)


class TestSticky:
    """Test that sticky rule works correctly."""

    def test_sticky_extends_alert(self):
        """sticky=3: after k-rule fires, alert stays for 3 more hours."""
        df = _make_df([0.8, 0.3, 0.1, 0.1, 0.1, 0.1])
        alerts = apply_policy(df, "prob", threshold=0.5, k=1, sticky=3)
        # h0: above → fire, sticky covers h1,h2,h3
        # h1: below but sticky → 1
        # h2: below but sticky → 1
        # h3: below but sticky → 1
        # h4: beyond sticky → 0
        # h5: → 0
        expected = [1, 1, 1, 1, 0, 0]
        np.testing.assert_array_equal(alerts, expected)

    def test_sticky_zero_no_extension(self):
        """sticky=0: no extension after k-rule fires."""
        df = _make_df([0.8, 0.3, 0.8, 0.3])
        alerts = apply_policy(df, "prob", threshold=0.5, k=1, sticky=0)
        expected = [1, 0, 1, 0]
        np.testing.assert_array_equal(alerts, expected)


class TestCausality:
    """Alerts never depend on future hours."""

    def test_prefix_equals_full(self):
        """Alerts on a prefix must equal the prefix of alerts on full series."""
        probs = [0.8, 0.6, 0.2, 0.9, 0.7, 0.3, 0.8, 0.8]
        df_full = _make_df(probs)
        alerts_full = apply_policy(df_full, "prob", threshold=0.5, k=2, sticky=1)

        for prefix_len in range(1, len(probs)):
            df_prefix = _make_df(probs[:prefix_len])
            alerts_prefix = apply_policy(df_prefix, "prob", threshold=0.5, k=2, sticky=1)
            np.testing.assert_array_equal(
                alerts_prefix,
                alerts_full[:prefix_len],
                err_msg=f"Prefix length {prefix_len} differs from full series",
            )


class TestMultiplePatients:
    """apply_policy handles multiple patients independently."""

    def test_two_patients(self):
        """Two patients in same df are processed independently."""
        df1 = _make_df([0.8, 0.3, 0.1], patient_id="P1")
        df2 = _make_df([0.1, 0.8, 0.8], patient_id="P2")
        df = pd.concat([df1, df2], ignore_index=True)
        alerts = apply_policy(df, "prob", threshold=0.5, k=1, sticky=0)
        # P1: [1, 0, 0], P2: [0, 1, 1]
        expected = [1, 0, 0, 0, 1, 1]
        np.testing.assert_array_equal(alerts, expected)
