"""Tests for sepsis.splits — acceptance checks per PLAN.md §P2.T1."""

import hashlib
from pathlib import Path

import pytest

from sepsis.config import load_config
from sepsis.splits import create_splits


@pytest.fixture(scope="module")
def split_data():
    """Run create_splits once and return (lockbox_df, folds_df)."""
    cfg = load_config("configs/config.yaml")
    lockbox_df, folds_df = create_splits(cfg)
    return lockbox_df, folds_df, cfg


def test_no_patient_in_dev_and_test(split_data):
    """No patient appears in both dev and test."""
    lockbox_df, _, _ = split_data
    dev_ids = set(lockbox_df[lockbox_df["split"] == "dev"]["patient_id"])
    test_ids = set(lockbox_df[lockbox_df["split"] == "test"]["patient_id"])
    assert len(dev_ids & test_ids) == 0, "Overlap between dev and test patients"


def test_no_patient_in_two_folds(split_data):
    """No patient appears in two different folds."""
    _, folds_df, _ = split_data
    dup = folds_df.groupby("patient_id")["fold"].nunique()
    assert (dup == 1).all(), "Some patients appear in multiple folds"


def test_septic_rate_per_fold(split_data):
    """Septic rate per fold within ±0.5 percentage points of overall."""
    _, folds_df, _ = split_data
    overall_rate = folds_df["septic"].mean() * 100.0
    for fold in range(5):
        fold_rate = folds_df[folds_df["fold"] == fold]["septic"].mean() * 100.0
        assert abs(fold_rate - overall_rate) < 0.5, (
            f"Fold {fold} septic rate {fold_rate:.2f}% differs from "
            f"overall {overall_rate:.2f}% by more than ±0.5pp"
        )


def test_hospital_share_per_fold(split_data):
    """Hospital share per fold within ±1 percentage point of overall."""
    _, folds_df, _ = split_data
    overall_a_rate = (folds_df["hospital"] == "A").mean() * 100.0
    for fold in range(5):
        fold_sub = folds_df[folds_df["fold"] == fold]
        fold_a_rate = (fold_sub["hospital"] == "A").mean() * 100.0
        assert abs(fold_a_rate - overall_a_rate) < 1.0, (
            f"Fold {fold} hospital A share {fold_a_rate:.2f}% differs from "
            f"overall {overall_a_rate:.2f}% by more than ±1pp"
        )


def test_reproducibility(split_data):
    """Re-running produces identical files (hash compare)."""
    _, _, cfg = split_data
    lockbox_path = Path(cfg["paths"]["lockbox"])
    folds_path = Path(cfg["paths"]["folds"])

    hash1_lockbox = hashlib.sha256(lockbox_path.read_bytes()).hexdigest()
    hash1_folds = hashlib.sha256(folds_path.read_bytes()).hexdigest()

    # Re-run
    create_splits(cfg)

    hash2_lockbox = hashlib.sha256(lockbox_path.read_bytes()).hexdigest()
    hash2_folds = hashlib.sha256(folds_path.read_bytes()).hexdigest()

    assert hash1_lockbox == hash2_lockbox, "lockbox.csv changed on re-run"
    assert hash1_folds == hash2_folds, "folds.csv changed on re-run"


def test_all_dev_patients_have_fold(split_data):
    """All dev patients should have a fold assignment (0-4)."""
    _, folds_df, _ = split_data
    assert (folds_df["fold"] >= 0).all()
    assert (folds_df["fold"] <= 4).all()
    assert set(folds_df["fold"].unique()) == {0, 1, 2, 3, 4}


def test_cross_hospital_folds_exist(split_data):
    """All dev patients should have a fold_within_hospital assignment."""
    _, folds_df, _ = split_data
    assert (folds_df["fold_within_hospital"] >= 0).all()
    assert (folds_df["fold_within_hospital"] <= 4).all()
