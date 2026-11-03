"""API tests for SepsisSentinel backend (P7.T2; both served model kinds since P9)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.settings import Settings, get_settings
from sepsis.constants import LABS, RAW_COLUMNS, VITALS

settings = get_settings()
POLICY_KEYS = ("threshold", "k", "sticky")
FROZEN_POLICY = {"threshold": 0.03, "k": 1, "sticky": 0}
OUTCOMES = {"septic_early", "septic_missed_or_late", "septic_other", "nonseptic_quiet",
            "nonseptic_false_alert"}
pytestmark = pytest.mark.skipif(
    not (settings.demo_dir / "demo_predictions_lgbm.parquet").exists()
    or not (settings.ensemble_dir / "ensemble.json").exists(),
    reason="demo cohort / ensemble not built (python -m sepsis.demo_cohort --rescore)",
)


@pytest.fixture(scope="module", params=["ensemble", "lgbm"])
def client(request):
    with TestClient(create_app(Settings(model_kind=request.param))) as c:
        c.kind = request.param
        yield c


@pytest.fixture(scope="module")
def demo():
    patients = pd.read_csv(settings.demo_dir / "demo_patients.csv")
    preds = {
        "ensemble": pd.read_parquet(settings.demo_dir / "demo_predictions.parquet"),
        "lgbm": pd.read_parquet(settings.demo_dir / "demo_predictions_lgbm.parquet"),
    }
    rows = pd.read_parquet(settings.demo_dir / "demo_rows.parquet")
    return patients, preds, rows


def expected_policy(kind: str) -> dict:
    if kind == "ensemble":
        with open(settings.ensemble_dir / "ensemble.json") as f:
            return json.load(f)["policy"]
    with open(settings.model_dir / "policy.json") as f:
        return json.load(f)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_kind"] == client.kind
    assert "lgbm-final-303r" in body["model_version"]
    assert body["model_version"].startswith("ensemble-" if client.kind == "ensemble" else "lgbm-")


def test_config_matches_policy(client):
    cfg = client.get("/api/config").json()
    policy = expected_policy(client.kind)
    assert cfg["threshold"] == pytest.approx(policy["threshold"])
    assert (cfg["k"], cfg["sticky"]) == (policy["k"], policy["sticky"])
    assert cfg["n_patients"] == 24
    assert cfg["model_kind"] == client.kind
    assert "LightGBM" in cfg["reasons_source"]
    assert (cfg["weight_lgbm"] is None) == (client.kind == "lgbm")
    assert "Not a medical device" in cfg["disclaimer"]


def test_lgbm_policy_is_frozen():
    policy = expected_policy("lgbm")
    assert {k: policy[k] for k in POLICY_KEYS} == FROZEN_POLICY


def test_demo_cohort_composition(demo):
    patients, _, _ = demo
    counts = patients["outcome"].value_counts().to_dict()
    assert counts == {
        "septic_early": 8,
        "septic_missed_or_late": 2,
        "nonseptic_quiet": 10,
        "nonseptic_false_alert": 4,
    }
    assert set(patients["outcome_ensemble"]) <= OUTCOMES
    assert (patients["septic"] == 1).sum() == 10
    assert patients["stay_hours"].between(24, 96).all()
    assert list(patients["display_name"]) == [f"Bed {i:02d}" for i in range(1, 25)]


def test_ward_sorted_by_risk(client):
    beds = client.get("/api/ward", params={"t": 0}).json()
    assert len(beds) == 24
    risks = [b["risk"] for b in beds]
    assert risks == sorted(risks, reverse=True)
    assert all(b["status"] == "active" for b in beds)


def test_ward_levels_follow_threshold(client):
    tau = 0.03
    for b in client.get("/api/ward", params={"t": 20, "threshold": tau}).json():
        if b["alert"]:
            assert b["level"] == "ALERT" and b["risk"] >= tau
        elif b["risk"] >= 0.5 * tau:
            assert b["level"] == "WATCH"
        else:
            assert b["level"] == "OK"
        assert len(b["risk_trend"]) <= 12


def test_t_beyond_stay_is_discharged(client, demo):
    patients, _, _ = demo
    t = int(patients["stay_hours"].min())
    beds = client.get("/api/ward", params={"t": t}).json()
    shortest = patients.loc[patients["stay_hours"].idxmin(), "patient_id"]
    bed = next(b for b in beds if b["patient_id"] == shortest)
    assert bed["status"] == "discharged"
    assert bed["hour"] == t - 1
    statuses = [b["status"] for b in beds]
    assert statuses == sorted(statuses, key=lambda s: s == "discharged")


def test_patient_view_and_truth(client, demo):
    patients, _, _ = demo
    pid = patients.loc[patients["outcome"] == "septic_early", "patient_id"].iloc[0]
    hidden = client.get(f"/api/patients/{pid}", params={"t": 10}).json()
    assert hidden["onset_hour"] is None and hidden["septic"] is None
    assert hidden["hours"] == list(range(11))
    assert len(hidden["risk"]) == len(hidden["alert"]) == len(hidden["vitals"]["HR"]) == 11
    assert len(hidden["reasons"]) <= 5
    shown = client.get(f"/api/patients/{pid}", params={"t": 10, "show_truth": True}).json()
    row = patients.set_index("patient_id").loc[pid]
    assert shown["onset_hour"] == int(row["t_sepsis"])
    col = "outcome_ensemble" if client.kind == "ensemble" else "outcome"
    assert shown["outcome"] == row[col]


def test_patient_view_has_every_measurement_and_hourly_insights(client, demo):
    patients, preds, rows = demo
    pid = patients.loc[patients["outcome"] == "septic_early", "patient_id"].iloc[0]
    view = client.get(f"/api/patients/{pid}", params={"t": 12}).json()
    assert set(view["vitals"]) == set(VITALS) and set(view["labs"]) == set(LABS)
    assert all(len(v) == 13 for v in [*view["vitals"].values(), *view["labs"].values()])
    assert len(view["reasons_by_hour"]) == 13 and view["reasons_by_hour"][-1] == view["reasons"]
    stored = preds[client.kind]
    stored = stored[stored["patient_id"] == pid].sort_values("hour")
    assert view["reasons_by_hour"][5] == json.loads(stored["top_reasons_json"].iloc[5])
    raw = rows[rows["patient_id"] == pid].sort_values("hour")
    hr = [None if math.isnan(x) else x for x in raw["HR"].head(13)]
    assert view["vitals"]["HR"] == pytest.approx(hr, nan_ok=True)
    if client.kind == "ensemble":
        comp = view["risk_components"]
        assert len(comp["lgbm"]) == len(comp["gru"]) == 13
        assert comp["lgbm"] == pytest.approx(stored["risk_lgbm"].head(13).tolist())
    else:
        assert view["risk_components"] is None


def test_patient_404_and_422(client):
    assert client.get("/api/patients/nobody").status_code == 404
    assert client.get("/api/ward", params={"t": -1}).status_code == 422
    assert client.get("/api/ward", params={"threshold": 1.5}).status_code == 422


def test_threshold_changes_alerts(client):
    low = client.get("/api/ward", params={"t": 20, "threshold": 0.01}).json()
    high = client.get("/api/ward", params={"t": 20, "threshold": 0.9}).json()
    n_low = sum(b["alert"] for b in low)
    n_high = sum(b["alert"] for b in high)
    assert n_low > n_high


def test_predict_consistency(client, demo):
    """POST /api/predict on a demo patient's raw rows reproduces the stored risks."""
    _, preds, rows = demo
    pid = rows["patient_id"].iloc[0]
    prow = rows[rows["patient_id"] == pid].sort_values("hour")
    payload = []
    for _, r in prow.iterrows():
        d = {"hour": int(r["hour"])}
        for c in RAW_COLUMNS:
            v = float(r[c])
            d[c] = None if math.isnan(v) else v
        payload.append(d)
    r = client.post("/api/predict", json={"rows": payload})
    assert r.status_code == 200
    out = r.json()
    assert out["model_kind"] == client.kind
    expected = preds[client.kind]
    expected = expected[expected["patient_id"] == pid].sort_values("hour")
    assert np.abs(np.array(out["risk"]) - expected["risk"].to_numpy()).max() < 1e-6
    assert out["alert"] == [bool(a) for a in expected["alert"]]
    assert len(out["reasons_last_hour"]) <= 5
    assert "LightGBM" in out["reasons_source"]


def test_predict_rejects_bad_body(client):
    assert client.post("/api/predict", json={"rows": []}).status_code == 422
    assert client.post("/api/predict", json={"rows": [{"hour": 0, "Bogus": 1}]}).status_code == 422


def test_metrics_are_frozen_values(client):
    m = client.get("/api/metrics").json()
    assert m["active_model"] == client.kind
    lockbox = m["lockbox_test"]
    assert lockbox["label"] == "Lockbox test (LightGBM)"
    test = {r["metric"]: r["value"] for r in lockbox["metrics"]}
    assert test["utility_norm"] == pytest.approx(0.42695526542981604)
    assert test["auroc"] == pytest.approx(0.8475067174931725)
    assert len(lockbox["baselines"]) == 4
    assert lockbox["policy"]["threshold"] == pytest.approx(0.03)
    assert len(m["cross_hospital"]) == 16
    assert m["policy"]["threshold"] == pytest.approx(expected_policy(client.kind)["threshold"])


def test_metrics_dev_cv_section(client):
    dev = client.get("/api/metrics").json()["dev_cv"]
    assert dev["label"] == "Development comparison (nested 5-fold CV)"
    table = pd.read_csv(settings.tables_dir / "ensemble_dev_comparison.csv")
    nested = table[(table["model"] == "lgbm") & (table["scheme"] == "nested")].iloc[0]
    row = next(r for r in dev["comparison"] if r["model"] == "lgbm" and r["scheme"] == "nested")
    assert row["utility_norm"] == pytest.approx(nested["utility_norm"])
    assert {r["weight_lgbm"] for r in dev["weights"]} == {0.5, 0.6, 0.7, 0.8, 0.9}
    assert len(dev["final"]) == 2 and dev["bootstrap"]


def test_model_card(client):
    card = client.get("/api/model-card").json()
    assert card["model_kind"] == client.kind
    assert "Intended use" in card["sections"]
    assert "0.4270" in card["markdown"]


def test_figures(client):
    r = client.get("/api/figures/fig_calibration_test.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/figures/..%2Fmodel_card.md").status_code == 404
