"""Unit tests for sepsis.explain — P5.T3."""

import json
from pathlib import Path

import numpy as np

from sepsis.explain import feature_text


def test_feature_text_all_features_non_empty():
    """Verify that feature_text returns non-empty string for every feature."""
    feature_list_path = Path("models/final/feature_list.json")
    assert feature_list_path.exists(), "feature_list.json must exist"

    with open(feature_list_path) as f:
        feature_list = json.load(f)

    assert len(feature_list) > 0

    # Test with standard positive, negative, zero, and nan values
    test_values = [0.0, 1.0, 42.5, -3.2, np.nan]

    for feat in feature_list:
        for val in test_values:
            txt = feature_text(feat, val)
            assert isinstance(txt, str)
            assert len(txt.strip()) > 0, f"Empty text for feature '{feat}' with value {val}"


def test_feature_text_clinical_patterns():
    """Test specific expected patterns for clinical readability."""
    assert "Heart rate" in feature_text("HR", 85.0)
    assert "rising" in feature_text("HR__slope_6h", 1.5)
    assert "falling" in feature_text("HR__slope_6h", -1.5)
    assert "measured this hour" in feature_text("WBC__meas", 1.0)
    assert "last measured 5h ago" in feature_text("Lactate__tsl", 5.0)
    assert "Shock index 0.85" in feature_text("idx__shock_index", 0.85)
    assert "SIRS criteria met: 3/4" in feature_text("score__sirs", 3.0)
    assert "Partial SOFA score 4/13" in feature_text("score__sofa_partial", 4.0)
    assert "Age 65 years" in feature_text("static__age", 65.0)
