"""Demo cohort for the dashboard — P7.T1.

From lockbox test patients with stay 24–96 h, after applying the frozen final model + policy,
choose (seed 42): 8 septic detected early (lead 4–12 h), 2 septic missed or late, 10 non-septic
without alerts, 4 non-septic with >= 1 false alert. No metrics are computed here; the lockbox
evaluation in P5.T6 is untouched.

`--rescore` (P9) keeps the same 24 beds and scores them with both LightGBM and the dev-only
ensemble, so the app can switch. Bed outcomes are recomputed per model with the same rules.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED
from sepsis.inference import (
    Artifacts,
    EnsembleArtifacts,
    alerts_for,
    gru_inputs,
    load_artifacts,
    load_ensemble,
    reasons_for,
    score_clean_rows,
    score_clean_rows_ensemble,
)

logger = logging.getLogger(__name__)

COHORT_SPEC = {
    "septic_early": 8,
    "septic_missed_or_late": 2,
    "nonseptic_quiet": 10,
    "nonseptic_false_alert": 4,
}
MIN_STAY, MAX_STAY = 24, 96
EARLY_LEAD = (4.0, 12.0)
ONLINE_BATCH_TOL = 1e-5


def classify_patient(labels: np.ndarray, alerts: np.ndarray) -> Dict[str, Any]:
    """Return septic flag, t_sepsis, lead time to first alert and demo outcome category."""
    alert_hours = np.flatnonzero(alerts)
    first_alert = int(alert_hours[0]) if len(alert_hours) else None
    if labels.any():
        t_sepsis = int(np.argmax(labels)) + 6
        lead = float(t_sepsis - first_alert) if first_alert is not None else None
        if lead is not None and EARLY_LEAD[0] <= lead <= EARLY_LEAD[1]:
            outcome = "septic_early"
        elif lead is None or lead <= 0:
            outcome = "septic_missed_or_late"
        else:
            outcome = "septic_other"
        return {"septic": 1, "t_sepsis": t_sepsis, "lead_time_h": lead, "outcome": outcome}
    outcome = "nonseptic_false_alert" if first_alert is not None else "nonseptic_quiet"
    return {"septic": 0, "t_sepsis": None, "lead_time_h": None, "outcome": outcome}


def build_demo_cohort(cfg: Dict[str, Any], out_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Select the 24-bed demo cohort and write rows, predictions and patient table."""
    set_seeds(SEED)
    art = load_artifacts(model_dir)
    policy = art.policy
    tau = float(policy["threshold"])

    lockbox = pd.read_csv(cfg["paths"]["lockbox"])
    test_pids = set(lockbox.loc[lockbox["split"] == "test", "patient_id"])
    clean_rows = pd.read_parquet(cfg["paths"]["clean_rows"])
    clean_rows = clean_rows[clean_rows["patient_id"].isin(test_pids)]
    stay = clean_rows.groupby("patient_id").size()
    cand = sorted(stay[(stay >= MIN_STAY) & (stay <= MAX_STAY)].index)
    logger.info("Candidates: %d test patients with stay %d–%d h", len(cand), MIN_STAY, MAX_STAY)

    # Screen candidates with the precomputed causal features (identical code path)
    feats = pd.read_parquet(cfg["paths"]["features"])
    feats = feats[feats["patient_id"].isin(set(cand))].sort_values(["patient_id", "hour"])
    raw_prob = art.booster.predict(feats[art.feature_list])
    screen = feats[["patient_id", "hour", "SepsisLabel"]].reset_index(drop=True)
    screen["risk"] = np.clip(art.calibrator.predict(raw_prob), 0.0, 1.0)
    screen["alert"] = alerts_for(screen, policy, tau)

    info = []
    for pid, g in screen.groupby("patient_id", sort=True):
        row = classify_patient(g["SepsisLabel"].to_numpy(int), g["alert"].to_numpy(int))
        info.append({"patient_id": pid, **row})
    info_df = pd.DataFrame(info)
    logger.info("Outcome counts among candidates:\n%s", info_df["outcome"].value_counts())

    rng = np.random.RandomState(SEED)
    chosen = []
    for outcome, n in COHORT_SPEC.items():
        pool = sorted(info_df.loc[info_df["outcome"] == outcome, "patient_id"])
        if len(pool) < n:
            raise ValueError(f"Only {len(pool)} candidates for {outcome}, need {n}")
        chosen.extend(rng.choice(pool, size=n, replace=False).tolist())

    demo_clean = clean_rows[clean_rows["patient_id"].isin(set(chosen))]
    demo_clean = demo_clean.sort_values(["patient_id", "hour"]).reset_index(drop=True)

    # Re-score the 24 patients through the online path the API uses
    demo_feats, scores = score_clean_rows(demo_clean, art, cfg)
    scores["alert"] = alerts_for(scores, policy, tau)
    ref = screen[screen["patient_id"].isin(set(chosen))].sort_values(["patient_id", "hour"])
    max_diff = float(np.abs(ref["risk"].to_numpy() - scores["risk"].to_numpy()).max())
    logger.info("Online vs batch risk max abs diff: %.2e", max_diff)
    if max_diff > 1e-6:
        raise AssertionError(f"Online path diverges from batch features ({max_diff:.2e})")

    reasons = reasons_for(demo_feats, art, top_n=5)
    preds = scores[["patient_id", "hour", "risk"]].copy()
    preds["alert"] = scores["alert"].astype("int8")
    preds["top_reasons_json"] = [json.dumps(r) for r in reasons]

    # Patient table, bed numbers shuffled with seed 42
    first = demo_clean.groupby("patient_id").first()
    patients = info_df[info_df["patient_id"].isin(set(chosen))].set_index("patient_id")
    patients = patients.loc[sorted(chosen)].reset_index()
    order = np.random.RandomState(SEED).permutation(len(patients))
    patients = patients.iloc[order].reset_index(drop=True)
    patients.insert(1, "display_name", [f"Bed {i + 1:02d}" for i in range(len(patients))])
    patients["age"] = patients["patient_id"].map(first["Age"]).round(0).astype(int)
    patients["gender"] = patients["patient_id"].map(first["Gender"]).astype(int)
    patients["unit"] = patients["patient_id"].map(first["unit"]).astype(int)
    patients["stay_hours"] = patients["patient_id"].map(stay).astype(int)
    patients["t_sepsis"] = patients["t_sepsis"].astype("Int64")
    cols = [
        "patient_id",
        "display_name",
        "age",
        "gender",
        "unit",
        "stay_hours",
        "septic",
        "t_sepsis",
        "lead_time_h",
        "outcome",
    ]
    patients = patients[cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    demo_clean.to_parquet(out_dir / "demo_rows.parquet", index=False)
    preds.to_parquet(out_dir / "demo_predictions.parquet", index=False)
    patients.to_csv(out_dir / "demo_patients.csv", index=False)
    logger.info(
        "Saved demo cohort: %d patients, %d rows -> %s", len(patients), len(demo_clean), out_dir
    )
    return patients


def _outcomes(scores: pd.DataFrame, alerts: np.ndarray) -> pd.DataFrame:
    """Per-bed outcome category and lead time from one model's alerts."""
    df = scores[["patient_id", "SepsisLabel"]].assign(alert=alerts)
    rows = [
        {
            "patient_id": pid,
            **classify_patient(g["SepsisLabel"].to_numpy(int), g["alert"].to_numpy(int)),
        }
        for pid, g in df.groupby("patient_id", sort=True)
    ]
    return pd.DataFrame(rows).set_index("patient_id")


def batch_ensemble_risk(
    cfg: Dict[str, Any], pids: List[str], art: Artifacts, ens: EnsembleArtifacts
) -> pd.DataFrame:
    """Ensemble risk from the precomputed feature table with a padded, batched GRU pass."""
    import torch

    from sepsis.train_gru import predict

    feats = pd.read_parquet(cfg["paths"]["features"], filters=[("patient_id", "in", pids)])
    feats = feats.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    risk_l = np.clip(art.calibrator.predict(art.booster.predict(feats[art.feature_list])), 0, 1)
    X = gru_inputs(feats, ens)
    p = feats["patient_id"].to_numpy()
    offsets = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1, len(feats)]
    patients = np.arange(len(offsets) - 1)
    raw = predict(ens.gru, X, np.zeros(len(feats)), offsets, patients, torch.device("cpu"))
    risk_g = np.clip(ens.gru_calibrator.predict(raw), 0, 1)
    out = feats[["patient_id", "hour"]].copy()
    out["risk"] = ens.weight * risk_l + (1 - ens.weight) * risk_g
    return out


def rescore_demo_cohort(
    cfg: Dict[str, Any], demo_dir: Path, model_dir: Path, ens_dir: Path
) -> pd.DataFrame:
    """Score the existing 24 beds (not reselected) with LightGBM and with the ensemble."""
    art = load_artifacts(model_dir)
    ens = load_ensemble(ens_dir)
    patients = pd.read_csv(demo_dir / "demo_patients.csv", dtype={"patient_id": str})
    demo_clean = pd.read_parquet(demo_dir / "demo_rows.parquet")
    demo_clean = demo_clean.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    pids = sorted(patients["patient_id"])
    if len(pids) != 24 or set(pids) != set(demo_clean["patient_id"]):
        raise ValueError("demo_rows.parquet and demo_patients.csv disagree on the 24 beds")

    feats, scores = score_clean_rows_ensemble(demo_clean, art, ens, cfg)
    reasons = [json.dumps(r) for r in reasons_for(feats, art, top_n=5)]

    # LightGBM alone: the frozen model and its policy, unchanged
    lgbm = scores[["patient_id", "hour"]].assign(risk=scores["risk_lgbm"])
    lgbm["alert"] = alerts_for(lgbm, art.policy, float(art.policy["threshold"])).astype("int8")
    lgbm["top_reasons_json"] = reasons
    ens_path = demo_dir / "demo_predictions.parquet"
    if ens_path.exists():
        old = pd.read_parquet(ens_path)
        if "risk_gru" not in old.columns:  # still the original LightGBM file
            old = old.sort_values(["patient_id", "hour"])
            gap = float(np.abs(old["risk"].to_numpy() - lgbm["risk"].to_numpy()).max())
            logger.info("LightGBM risks vs the original demo file: max abs diff %.2e", gap)
            if gap > 1e-6:
                raise AssertionError(f"Frozen LightGBM no longer reproduces the demo ({gap:.2e})")

    batch = batch_ensemble_risk(cfg, pids, art, ens)
    max_diff = float(np.abs(batch["risk"].to_numpy() - scores["risk"].to_numpy()).max())
    logger.info("Ensemble online vs batch risk max abs diff: %.2e", max_diff)
    if max_diff > ONLINE_BATCH_TOL:
        raise AssertionError(f"Ensemble online path diverges from batch ({max_diff:.2e})")

    ens_preds = scores[["patient_id", "hour", "risk"]].copy()
    tau = float(ens.policy["threshold"])
    ens_preds["alert"] = alerts_for(ens_preds, ens.policy, tau).astype("int8")
    ens_preds["top_reasons_json"] = reasons
    ens_preds["risk_lgbm"] = scores["risk_lgbm"]
    ens_preds["risk_gru"] = scores["risk_gru"]

    out_l = _outcomes(scores, lgbm["alert"].to_numpy())
    out_e = _outcomes(scores, ens_preds["alert"].to_numpy())
    idx = patients["patient_id"]
    if not (out_l.loc[idx, "outcome"].to_numpy() == patients["outcome"].to_numpy()).all():
        raise AssertionError("LightGBM bed outcomes changed")
    patients["lead_time_h_ensemble"] = out_e.loc[idx, "lead_time_h"].to_numpy()
    patients["outcome_ensemble"] = out_e.loc[idx, "outcome"].to_numpy()

    lgbm.to_parquet(demo_dir / "demo_predictions_lgbm.parquet", index=False)
    ens_preds.to_parquet(ens_path, index=False)
    patients.to_csv(demo_dir / "demo_patients.csv", index=False)
    summary = {
        "online_vs_batch_max_abs_diff_ensemble": max_diff,
        "tolerance": ONLINE_BATCH_TOL,
        "ensemble_policy": ens.policy,
        "weight_lgbm": ens.weight,
        "outcomes_lgbm": patients["outcome"].value_counts().to_dict(),
        "outcomes_ensemble": patients["outcome_ensemble"].value_counts().to_dict(),
    }
    with open(demo_dir / "demo_scoring.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Ensemble bed outcomes: %s", summary["outcomes_ensemble"])
    return patients


def main() -> None:
    """CLI entrypoint: python -m sepsis.demo_cohort [--rescore]."""
    parser = argparse.ArgumentParser(description="Build the 24-bed dashboard demo cohort")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--out", type=str, default="data/demo")
    parser.add_argument("--model-dir", type=str, default="models/final")
    parser.add_argument("--ensemble-dir", type=str, default="models/ensemble_final")
    parser.add_argument(
        "--rescore", action="store_true", help="Score the existing 24 beds with both models"
    )
    args = parser.parse_args()
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    cfg = load_config(args.config)
    if args.rescore:
        patients = rescore_demo_cohort(
            cfg, Path(args.out), Path(args.model_dir), Path(args.ensemble_dir)
        )
    else:
        patients = build_demo_cohort(cfg, Path(args.out), Path(args.model_dir))
    print(patients.to_string(index=False))


if __name__ == "__main__":
    main()
