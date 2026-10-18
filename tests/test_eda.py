"""Unit tests for sepsis.eda module."""

import numpy as np
import pandas as pd

from sepsis.constants import MEASURED
from sepsis.eda import generate_dataset_summary, generate_ks_tests, generate_missingness_by_var


def test_generate_dataset_summary_synthetic():
    """Test dataset summary computation on a small synthetic DataFrame."""
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p3", "p3"],
            "hospital": ["A", "A", "A", "B", "B", "B"],
            "hour": [0, 1, 0, 0, 1, 2],
            "Age": [60.0, 60.0, 70.0, 50.0, 50.0, 50.0],
            "Gender": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "SepsisLabel": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    summary_df = generate_dataset_summary(df)
    assert len(summary_df) == 3  # A, B, Total
    tot = summary_df[summary_df["hospital"] == "Total"].iloc[0]
    assert tot["patients"] == 3
    assert tot["rows"] == 6
    assert tot["septic_patients"] == 1
    assert tot["positive_rows"] == 1


def test_generate_missingness_and_ks():
    """Test missingness calculation and KS test on synthetic data."""
    data = {
        "patient_id": ["p1", "p2"],
        "hospital": ["A", "B"],
        "hour": [0, 0],
    }
    for var in MEASURED:
        data[var] = [np.nan, 10.0]

    df = pd.DataFrame(data)
    missing_df = generate_missingness_by_var(df)
    assert len(missing_df) == len(MEASURED)
    assert missing_df.loc[0, "pct_missing_overall"] == 50.0
    assert missing_df.loc[0, "pct_missing_A"] == 100.0
    assert missing_df.loc[0, "pct_missing_B"] == 0.0

    ks_df = generate_ks_tests(df)
    assert len(ks_df) == len(MEASURED)
