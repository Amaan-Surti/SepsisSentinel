"""Configuration loader and seed utilities for reproducibility."""

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from sepsis.constants import SEED


def load_config(path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load project YAML configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seeds(seed: int = SEED) -> None:
    """Set random seeds across python, numpy, and environment for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
