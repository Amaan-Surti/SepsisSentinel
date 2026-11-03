"""Backend settings: artifact paths with environment-variable overrides."""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DISCLAIMER = (
    "Research demonstration on retrospective ICU data. Not approved for clinical diagnostic use. "
    "Decision support on retrospective data. Not a medical device."
)


MODEL_KINDS = ("ensemble", "lgbm")


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default)))


def _model_kind() -> str:
    kind = os.environ.get("MODEL_KIND", "ensemble").strip().lower()
    if kind not in MODEL_KINDS:
        raise ValueError(f"MODEL_KIND must be one of {MODEL_KINDS}, got {kind!r}")
    return kind


@dataclass(frozen=True)
class Settings:
    """Served model kind plus locations of the models, demo cohort and report tables.

    MODEL_KIND=ensemble (default) serves the dev-chosen LightGBM + GRU ensemble;
    MODEL_KIND=lgbm serves the frozen LightGBM alone.
    """

    model_kind: str = field(default_factory=_model_kind)
    model_dir: Path = field(default_factory=lambda: _path("MODEL_DIR", REPO_ROOT / "models/final"))
    ensemble_dir: Path = field(
        default_factory=lambda: _path("ENSEMBLE_DIR", REPO_ROOT / "models/ensemble_final")
    )
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

    def __post_init__(self) -> None:
        if self.model_kind not in MODEL_KINDS:
            raise ValueError(f"model_kind must be one of {MODEL_KINDS}, got {self.model_kind!r}")


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings()
