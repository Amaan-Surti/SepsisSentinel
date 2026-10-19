"""Demo cohort for the dashboard — P7.T1.

From lockbox test patients with stay 24–96 h, after applying the frozen final model + policy,
choose (seed 42): 8 septic detected early (lead 4–12 h), 2 septic missed or late, 10 non-septic
without alerts, 4 non-septic with >= 1 false alert. No metrics are computed here; the lockbox
evaluation in P5.T6 is untouched.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED
from sepsis.inference import alerts_for, load_artifacts, reasons_for, score_clean_rows

logger = logging.getLogger(__name__)

COHORT_SPEC = {
    "septic_early": 8,
    "septic_missed_or_late": 2,
    "nonseptic_quiet": 10,
    "nonseptic_false_alert": 4,
}
MIN_STAY, MAX_STAY = 24, 96
EARLY_LEAD = (4.0, 12.0)


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


def main() -> None:
    """CLI entrypoint: python -m sepsis.demo_cohort."""
    parser = argparse.ArgumentParser(description="Build the 24-bed dashboard demo cohort")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--out", type=str, default="data/demo")
    parser.add_argument("--model-dir", type=str, default="models/final")
    args = parser.parse_args()
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    patients = build_demo_cohort(load_config(args.config), Path(args.out), Path(args.model_dir))
    print(patients.to_string(index=False))


if __name__ == "__main__":
    main()
