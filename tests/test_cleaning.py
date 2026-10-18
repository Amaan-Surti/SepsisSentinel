"""Unit tests for sepsis.cleaning module."""

import numpy as np
import pandas as pd

from sepsis.cleaning import clean


def test_clean_sanity_ranges():
    """Test values outside physiological ranges are set to NaN without clipping."""
    df = pd.DataFrame(
        {
            "HR": [10.0, 75.0, 310.0],  # Min 20, Max 300
            "Temp": [20.0, 37.0, 50.0],  # Min 25, Max 45
            "O2Sat": [40.0, 98.0, 105.0],  # Min 50, Max 100
            "Unit1": [1.0, np.nan, 0.0],
            "Unit2": [0.0, 1.0, np.nan],
        }
    )
    cleaned, report = clean(df)

    assert np.isnan(cleaned.loc[0, "HR"])
    assert cleaned.loc[1, "HR"] == 75.0
    assert np.isnan(cleaned.loc[2, "HR"])

    assert np.isnan(cleaned.loc[0, "Temp"])
    assert cleaned.loc[1, "Temp"] == 37.0
    assert np.isnan(cleaned.loc[2, "Temp"])

    assert np.isnan(cleaned.loc[0, "O2Sat"])
    assert cleaned.loc[1, "O2Sat"] == 98.0
    assert np.isnan(cleaned.loc[2, "O2Sat"])


def test_clean_fio2_special_logic():
    """Test FiO2 percentage conversion and [0.21, 1.0] thresholding."""
    df = pd.DataFrame(
        {
            "FiO2": [
                50.0,  # 50% -> 0.50
                100.0,  # 100% -> 1.0
                0.21,  # 0.21 -> 0.21
                0.15,  # < 0.21 -> NaN
                120.0,  # > 100% -> NaN
            ]
        }
    )
    cleaned, report = clean(df)

    assert np.isclose(cleaned.loc[0, "FiO2"], 0.50)
    assert np.isclose(cleaned.loc[1, "FiO2"], 1.00)
    assert np.isclose(cleaned.loc[2, "FiO2"], 0.21)
    assert np.isnan(cleaned.loc[3, "FiO2"])
    assert np.isnan(cleaned.loc[4, "FiO2"])


def test_clean_unit_encoding():
    """Test unit encoding: 1 for Unit1, 2 for Unit2, 0 otherwise."""
    df = pd.DataFrame(
        {
            "Unit1": [1.0, 0.0, np.nan, 0.0],
            "Unit2": [0.0, 1.0, np.nan, 0.0],
        }
    )
    cleaned, _ = clean(df)

    assert list(cleaned["unit"]) == [1, 2, 0, 0]
