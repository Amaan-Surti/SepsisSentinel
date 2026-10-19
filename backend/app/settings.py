"""Backend settings: artifact paths with environment-variable overrides."""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DISCLAIMER = (
    "Research demonstration on retrospective ICU data. Not approved for clinical diagnostic use. "
    "Decision support on retrospective data. Not a medical device."
)


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default)))


@dataclass(frozen=True)
class Settings:
    """Filesystem locations of the frozen model, demo cohort and report tables."""

    model_dir: Path = field(default_factory=lambda: _path("MODEL_DIR", REPO_ROOT / "models/final"))
    demo_dir: Path = field(default_factory=lambda: _path("DEMO_DIR", REPO_ROOT / "data/demo"))
    tables_dir: Path = field(
        default_factory=lambda: _path("TABLES_DIR", REPO_ROOT / "reports/tables")
    )
    figures_dir: Path = field(
        default_factory=lambda: _path("FIGURES_DIR", REPO_ROOT / "reports/figures")
    )
    model_card: Path = field(
        default_factory=lambda: _path("MODEL_CARD", REPO_ROOT / "reports/model_card.md")
    )
    config_path: Path = field(
        default_factory=lambda: _path("CONFIG_PATH", REPO_ROOT / "configs/config.yaml")
    )
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://localhost:8080")


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings()
