"""Smoke tests for SepsisSentinel package."""

import sepsis
from sepsis.config import load_config, set_seeds
from sepsis.constants import RAW_COLUMNS, SEED


def test_import_and_version():
    """Verify package imports and has a version."""
    assert sepsis.__version__ == "0.1.0"


def test_config_loader():
    """Verify configuration file loads and has expected keys."""
    cfg = load_config("configs/config.yaml")
    assert "seed" in cfg
    assert cfg["seed"] == SEED
    assert "paths" in cfg
    assert len(RAW_COLUMNS) == 41


def test_seed_setter():
    """Verify set_seeds executes without error."""
    set_seeds(42)
