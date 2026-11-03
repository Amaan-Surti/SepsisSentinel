"""Tests for the served LightGBM + GRU ensemble (P9)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sepsis.config import load_config
from sepsis.constants import RAW_COLUMNS
from sepsis.ensemble import WEIGHTS
from sepsis.ensemble_final import final_epochs
from sepsis.inference import (
    alerts_for,
    load_artifacts,
    prepare_raw_rows,
    score_clean_rows,
)

torch = pytest.importorskip("torch")

from sepsis.inference import load_ensemble, score_clean_rows_ensemble  # noqa: E402

POLICY_KEYS = ("threshold", "k", "sticky")
FROZEN_POLICY = {"threshold": 0.03, "k": 1, "sticky": 0}
ENS_DIR = Path("models/ensemble_final")
DEMO = Path("data/demo")
needs_artifacts = pytest.mark.skipif(
    not (ENS_DIR / "gru.pt").exists() or not (DEMO / "demo_predictions_lgbm.parquet").exists(),
    reason="ensemble / rescored demo cohort not built",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def art():
    return load_artifacts("models/final", with_explainer=False)


@pytest.fixture(scope="module")
def ens():
    return load_ensemble(ENS_DIR)


@pytest.fixture(scope="module")
def demo_rows():
    return pd.read_parquet(DEMO / "demo_rows.parquet").sort_values(["patient_id", "hour"])


def raw_stay(demo_rows: pd.DataFrame, pid: str) -> pd.DataFrame:
    """A demo bed as the API receives it: hour + raw columns, no labels."""
    rows = demo_rows[demo_rows["patient_id"] == pid].reset_index(drop=True)
    raw = rows[["hour"] + [c for c in RAW_COLUMNS if c != "SepsisLabel"]].copy()
    raw["SepsisLabel"] = np.nan
    return raw


def online_risk(raw: pd.DataFrame, art, ens, cfg) -> np.ndarray:
    _, scores = score_clean_rows_ensemble(prepare_raw_rows(raw), art, ens, cfg)
    return scores["risk"].to_numpy()


def test_final_epochs_is_median_best_plus_one():
    meta = {"folds": [{"best_epoch": e} for e in (2, 4, 2, 2, 4)]}
    assert final_epochs(meta) == 3


@needs_artifacts
def test_frozen_lgbm_still_gives_its_old_predictions(art, cfg, demo_rows):
    assert art.params["num_boost_round"] == 303
    assert {k: art.policy[k] for k in POLICY_KEYS} == FROZEN_POLICY
    _, scores = score_clean_rows(demo_rows.reset_index(drop=True), art, cfg)
    old = pd.read_parquet(DEMO / "demo_predictions_lgbm.parquet")
    old = old.sort_values(["patient_id", "hour"])
    assert np.abs(scores["risk"].to_numpy() - old["risk"].to_numpy()).max() < 1e-9
    alerts = alerts_for(scores, art.policy, 0.03)
    assert (np.asarray(alerts) == old["alert"].to_numpy()).all()


@needs_artifacts
def test_weight_and_policy_come_from_ensemble_json(ens):
    with open(ENS_DIR / "ensemble.json") as f:
        meta = json.load(f)
    assert ens.weight == meta["weight_lgbm"] and ens.weight in WEIGHTS
    assert ens.policy == meta["policy"]
    assert not next(ens.gru.parameters()).is_cuda and not ens.gru.training
    preds = pd.read_parquet(DEMO / "demo_predictions.parquet")
    blend = ens.weight * preds["risk_lgbm"] + (1 - ens.weight) * preds["risk_gru"]
    assert np.abs(blend.to_numpy() - preds["risk"].to_numpy()).max() < 1e-12
    alerts = alerts_for(preds.sort_values(["patient_id", "hour"]), ens.policy,
                        ens.policy["threshold"])
    assert (np.asarray(alerts) == preds.sort_values(["patient_id", "hour"])["alert"]).all()


@needs_artifacts
def test_ensemble_online_matches_stored_and_batch(art, ens, cfg, demo_rows):
    """Online path == the stored demo predictions; batch-vs-online gap recorded <= 1e-5."""
    _, scores = score_clean_rows_ensemble(demo_rows.reset_index(drop=True), art, ens, cfg)
    stored = pd.read_parquet(DEMO / "demo_predictions.parquet").sort_values(["patient_id", "hour"])
    assert np.abs(scores["risk"].to_numpy() - stored["risk"].to_numpy()).max() < 1e-6
    with open(DEMO / "demo_scoring.json") as f:
        assert json.load(f)["online_vs_batch_max_abs_diff_ensemble"] <= 1e-5
    if Path(cfg["paths"]["features"]).exists():
        from sepsis.demo_cohort import batch_ensemble_risk

        batch = batch_ensemble_risk(cfg, sorted(demo_rows["patient_id"].unique()), art, ens)
        assert np.abs(batch["risk"].to_numpy() - scores["risk"].to_numpy()).max() <= 1e-5


@needs_artifacts
def test_ensemble_online_path_is_causal(art, ens, cfg, demo_rows):
    """Truncating or changing later hours never changes an earlier ensemble risk."""
    patients = pd.read_csv(DEMO / "demo_patients.csv", dtype={"patient_id": str})
    pid = patients.loc[patients["septic"] == 1, "patient_id"].iloc[0]
    raw = raw_stay(demo_rows, pid)
    full = online_risk(raw, art, ens, cfg)
    for cut in (6, len(raw) // 2, len(raw) - 1):
        prefix = online_risk(raw.iloc[:cut], art, ens, cfg)
        assert np.abs(prefix - full[:cut]).max() < 1e-6
        changed = raw.copy()
        changed.loc[cut:, ["HR", "Resp"]] += 40.0
        changed.loc[cut:, "Lactate"] = 9.0
        changed.loc[cut:, "Temp"] = 39.5
        after = online_risk(changed, art, ens, cfg)
        assert np.abs(after[:cut] - full[:cut]).max() < 1e-6
        assert not np.allclose(after[cut:], full[cut:])
