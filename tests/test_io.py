"""Unit tests for sepsis.io module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from sepsis.constants import RAW_COLUMNS
from sepsis.io import read_patient


def test_read_patient_valid():
    """Test read_patient loads a real PSV file with correct columns."""
    sample_file = Path("data/raw/training_setA/p000001.psv")
    if sample_file.exists():
        df = read_patient(sample_file)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == RAW_COLUMNS
        assert len(df) > 0


def test_read_patient_invalid_columns():
    """Test read_patient raises ValueError when columns do not match RAW_COLUMNS."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".psv", delete=False) as tmp:
        tmp.write("HR|O2Sat|Temp\n80|98|37\n")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValueError, match="do not match expected RAW_COLUMNS"):
            read_patient(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
