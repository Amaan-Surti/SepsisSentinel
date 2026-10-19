"""API tests for SepsisSentinel backend (P7.T2)."""

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.settings import get_settings
from sepsis.constants import RAW_COLUMNS

settings = get_settings()
pytestmark = pytest.mark.skipif(
    not (settings.demo_dir / "demo_predictions.parquet").exists(),
    reason="demo cohort not built (python -m sepsis.demo_cohort)",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def demo():
    patients = pd.read_csv(settings.demo_dir / "demo_patients.csv")
    preds = pd.read_parquet(settings.demo_dir / "demo_predictions.parquet")
    rows = pd.read_parquet(settings.demo_dir / "demo_rows.parquet")
    return patients, preds, rows


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"].startswith("lgbm-final-303r")


def test_config_matches_policy(client):
    cfg = client.get("/api/config").json()
    assert cfg["threshold"] == pytest.approx(0.03)
    assert (cfg["k"], cfg["sticky"]) == (1, 0)
    assert cfg["n_patients"] == 24
    assert "Not a medical device" in cfg["disclaimer"]


def test_demo_cohort_composition(demo):
    patients, _, _ = demo
    counts = patients["outcome"].value_counts().to_dict()
    assert counts == {
        "septic_early": 8,
        "septic_missed_or_late": 2,
        "nonseptic_quiet": 10,
        "nonseptic_false_alert": 4,
    }
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
    t_sep = int(patients.set_index("patient_id").loc[pid, "t_sepsis"])
    assert shown["onset_hour"] == t_sep


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
    """POST /api/predict on a demo patient's raw rows reproduces stored risks."""
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
    expected = preds[preds["patient_id"] == pid].sort_values("hour")
    assert np.abs(np.array(out["risk"]) - expected["risk"].to_numpy()).max() < 1e-6
    assert out["alert"] == [bool(a) for a in expected["alert"]]
    assert len(out["reasons_last_hour"]) <= 5


def test_predict_rejects_bad_body(client):
    assert client.post("/api/predict", json={"rows": []}).status_code == 422
    assert client.post("/api/predict", json={"rows": [{"hour": 0, "Bogus": 1}]}).status_code == 422


def test_metrics_are_frozen_values(client):
    m = client.get("/api/metrics").json()
    test = {r["metric"]: r["value"] for r in m["test_metrics"]}
    assert test["utility_norm"] == pytest.approx(0.42695526542981604)
    assert test["auroc"] == pytest.approx(0.8475067174931725)
    assert len(m["baselines"]) == 4
    assert len(m["cross_hospital"]) == 16
    assert m["policy"]["threshold"] == pytest.approx(0.03)


def test_model_card(client):
    card = client.get("/api/model-card").json()
    assert "Intended use" in card["sections"]
    assert "0.4270" in card["markdown"]


def test_figures(client):
    r = client.get("/api/figures/fig_calibration_test.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/figures/..%2Fmodel_card.md").status_code == 404
