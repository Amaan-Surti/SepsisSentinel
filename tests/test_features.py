"""Tests for sepsis.features — acceptance checks per PLAN.md §P3.T1."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sepsis.config import load_config
from sepsis.constants import MEASURED, SEED
from sepsis.features import CARRY_COLS, FORBIDDEN_FEATURE_NAMES, _patient_features


def _make_test_patient(cfg=None):
    """Build a hand-crafted 8-hour patient for verification.

    HR = 80, 82, NaN, 90, NaN, NaN, 100, 101
    All other measured cols NaN. Static: Age=60, Gender=1, unit=0, HospAdmTime=-5, hour=0..7.
    """
    if cfg is None:
        cfg = load_config("configs/config.yaml")
    n = 8
    data = {col: [np.nan] * n for col in MEASURED}
    data["HR"] = [80.0, 82.0, np.nan, 90.0, np.nan, np.nan, 100.0, 101.0]
    data["Age"] = [60.0] * n
    data["Gender"] = [1.0] * n
    data["Unit1"] = [1.0] * n
    data["Unit2"] = [0.0] * n
    data["HospAdmTime"] = [-5.0] * n
    data["ICULOS"] = list(range(1, n + 1))
    data["SepsisLabel"] = [0.0] * n
    data["patient_id"] = ["test_p1"] * n
    data["hospital"] = ["A"] * n
    data["hour"] = list(range(n))
    data["unit"] = [0] * n

    df = pd.DataFrame(data)
    for col in MEASURED:
        df[col] = df[col].astype("float32")
    return df


class TestHandMadePatient:
    """Hand-made 8-hour patient with HR = 80, 82, NaN, 90, NaN, NaN, 100, 101."""

    @pytest.fixture(scope="class")
    def feats(self):
        cfg = load_config("configs/config.yaml")
        df = _make_test_patient(cfg)
        return _patient_features(df, cfg)

    def test_hr_locf(self, feats):
        """LOCF values: 80, 82, 82, 90, 90, 90, 100, 101."""
        expected = [80, 82, 82, 90, 90, 90, 100, 101]
        np.testing.assert_array_almost_equal(feats["HR"].values, expected)

    def test_hr_meas(self, feats):
        """Measured flag: 1,1,0,1,0,0,1,1."""
        expected = [1, 1, 0, 1, 0, 0, 1, 1]
        np.testing.assert_array_equal(feats["HR__meas"].values, expected)

    def test_hr_tsl(self, feats):
        """TSL: 0,0,1,0,1,2,0,0."""
        expected = [0, 0, 1, 0, 1, 2, 0, 0]
        np.testing.assert_array_almost_equal(feats["HR__tsl"].values, expected)

    def test_hr_mean_6h_at_t5(self, feats):
        """mean_6h at t=5: mean of observed in window [0..5] = mean(80,82,90) = 84.0."""
        # Window covers rows 0..5 (6 hours); observed: h0=80, h1=82, h3=90
        assert np.isclose(
            feats["HR__mean_6h"].iloc[5], 84.0, atol=0.01
        ), f"Expected 84.0, got {feats['HR__mean_6h'].iloc[5]}"

    def test_hr_slope_6h_at_t7(self, feats):
        """slope_6h at t=7: window rows 2..7, observed (h3=90, h6=100, h7=101).

        np.polyfit([3,6,7],[90,100,101],1)[0] should match.
        """
        expected_slope = np.polyfit([3, 6, 7], [90, 100, 101], 1)[0]
        actual = feats["HR__slope_6h"].iloc[7]
        assert np.isclose(
            actual, expected_slope, atol=0.01
        ), f"Expected slope ~{expected_slope:.4f}, got {actual}"


class TestClinicalScores:
    """Score checks on hand-made rows."""

    def test_sirs_score(self):
        """SIRS = 4 for Temp 39, HR 120, Resp 25, WBC 15."""
        cfg = load_config("configs/config.yaml")
        data = {col: [np.nan] for col in MEASURED}
        data["Temp"] = [39.0]
        data["HR"] = [120.0]
        data["Resp"] = [25.0]
        data["WBC"] = [15.0]
        data["Age"] = [50.0]
        data["Gender"] = [0.0]
        data["Unit1"] = [1.0]
        data["Unit2"] = [0.0]
        data["HospAdmTime"] = [0.0]
        data["ICULOS"] = [1.0]
        data["SepsisLabel"] = [0.0]
        data["patient_id"] = ["sirs_test"]
        data["hospital"] = ["A"]
        data["hour"] = [0]
        data["unit"] = [0]
        df = pd.DataFrame(data)
        for col in MEASURED:
            df[col] = df[col].astype("float32")
        feats = _patient_features(df, cfg)
        assert (
            feats["score__sirs"].iloc[0] == 4
        ), f"Expected SIRS=4, got {feats['score__sirs'].iloc[0]}"

    def test_sofa_partial_score(self):
        """SOFA partial: Platelets 40→3, Bilirubin 1.5→1, Creatinine 1.0→0, MAP 60→1 → total 5."""
        cfg = load_config("configs/config.yaml")
        data = {col: [np.nan] for col in MEASURED}
        data["Platelets"] = [40.0]
        data["Bilirubin_total"] = [1.5]
        data["Creatinine"] = [1.0]
        data["MAP"] = [60.0]
        data["Age"] = [50.0]
        data["Gender"] = [0.0]
        data["Unit1"] = [1.0]
        data["Unit2"] = [0.0]
        data["HospAdmTime"] = [0.0]
        data["ICULOS"] = [1.0]
        data["SepsisLabel"] = [0.0]
        data["patient_id"] = ["sofa_test"]
        data["hospital"] = ["A"]
        data["hour"] = [0]
        data["unit"] = [0]
        df = pd.DataFrame(data)
        for col in MEASURED:
            df[col] = df[col].astype("float32")
        feats = _patient_features(df, cfg)
        assert (
            feats["score__sofa_partial"].iloc[0] == 5.0
        ), f"Expected SOFA=5, got {feats['score__sofa_partial'].iloc[0]}"


class TestFeatureListJson:
    """feature_list.json must have no forbidden names."""

    def test_no_forbidden_names(self):
        feature_list_path = Path("models/feature_list.json")
        if not feature_list_path.exists():
            pytest.skip("feature_list.json not yet created (run build_features first)")
        with open(feature_list_path) as f:
            features = json.load(f)
        for forbidden in FORBIDDEN_FEATURE_NAMES:
            assert forbidden not in features, f"Forbidden name {forbidden} in feature_list.json"


class TestCausality:
    """Causality: features on prefix equal features on full stay for rows < k."""

    def test_prefix_causality(self):
        """For the hand-made patient, features at rows[:k] should equal full[:k]."""
        cfg = load_config("configs/config.yaml")
        df = _make_test_patient(cfg)
        full_feats = _patient_features(df, cfg)

        # Check prefix of length 4
        prefix_df = df.iloc[:4].copy()
        prefix_feats = _patient_features(prefix_df, cfg)

        # Get numeric feature columns only
        feat_cols = [c for c in full_feats.columns if c not in CARRY_COLS]
        for col in feat_cols:
            full_vals = full_feats[col].iloc[:4].values.astype(float)
            prefix_vals = prefix_feats[col].values.astype(float)
            np.testing.assert_allclose(
                prefix_vals,
                full_vals,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
                err_msg=f"Causality violation in column {col}",
            )

    def test_20_patients_causality(self):
        """For 20 random patients (seed 42), features on rows[:k] equal full[:k]."""
        clean_path = Path("data/interim/clean_rows.parquet")
        if not clean_path.exists():
            pytest.skip("data/interim/clean_rows.parquet does not exist")
        cfg = load_config("configs/config.yaml")
        clean_df = pd.read_parquet(clean_path)
        rng = np.random.RandomState(SEED)
        all_pids = clean_df["patient_id"].unique()
        sampled_pids = rng.choice(all_pids, size=20, replace=False)

        for pid in sampled_pids:
            pt_df = (
                clean_df[clean_df["patient_id"] == pid].sort_values("hour").reset_index(drop=True)
            )
            n_rows = len(pt_df)
            if n_rows < 4:
                continue
            k = rng.randint(2, n_rows)
            full_feats = _patient_features(pt_df, cfg)
            prefix_df = pt_df.iloc[:k].copy()
            prefix_feats = _patient_features(prefix_df, cfg)

            feat_cols = [c for c in full_feats.columns if c not in CARRY_COLS]
            for col in feat_cols:
                full_vals = full_feats[col].iloc[:k].values.astype(float)
                prefix_vals = prefix_feats[col].values.astype(float)
                np.testing.assert_allclose(
                    prefix_vals,
                    full_vals,
                    rtol=1e-4,
                    atol=1e-4,
                    equal_nan=True,
                    err_msg=f"Causality violation in patient {pid}, column {col}",
                )
