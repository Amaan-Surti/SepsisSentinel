"""Patient-level train/test splits and dev folds for SepsisSentinel."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from sepsis.config import load_config, set_seeds
from sepsis.constants import SEED

logger = logging.getLogger(__name__)


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-patient table with patient_id, hospital, septic."""
    pt = (
        df.groupby("patient_id")
        .agg(hospital=("hospital", "first"), septic=("SepsisLabel", "max"))
        .reset_index()
    )
    pt["septic"] = (pt["septic"] == 1.0).astype(int)
    return pt


def create_splits(
    cfg: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create lockbox split, dev folds, and cross-hospital folds."""
    if cfg is None:
        cfg = load_config()
    set_seeds(cfg.get("seed", SEED))

    all_rows_path = Path(cfg["paths"]["all_rows"])
    lockbox_path = Path(cfg["paths"]["lockbox"])
    folds_path = Path(cfg["paths"]["folds"])
    lockbox_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading all_rows from %s...", all_rows_path)
    df = pd.read_parquet(all_rows_path, columns=["patient_id", "hospital", "SepsisLabel"])
    pt = build_patient_table(df)
    logger.info("Patient table: %d patients", len(pt))

    # --- 1. Lockbox test set ---
    pt["stratum"] = pt["hospital"] + "_" + pt["septic"].astype(str)
    dev_pts, test_pts = train_test_split(
        pt,
        test_size=cfg["split"]["lockbox_frac"],
        stratify=pt["stratum"],
        random_state=SEED,
    )
    dev_pts = dev_pts.copy()
    test_pts = test_pts.copy()
    dev_pts["split"] = "dev"
    test_pts["split"] = "test"

    lockbox_df = pd.concat([dev_pts, test_pts], ignore_index=True)
    lockbox_df = lockbox_df[["patient_id", "hospital", "septic", "split"]]
    lockbox_df.to_csv(lockbox_path, index=False)
    logger.info(
        "Lockbox split saved to %s: %d dev, %d test",
        lockbox_path,
        len(dev_pts),
        len(test_pts),
    )

    # --- 2. Dev folds ---
    dev_only = lockbox_df[lockbox_df["split"] == "dev"].copy().reset_index(drop=True)
    dev_only["stratum"] = dev_only["hospital"] + "_" + dev_only["septic"].astype(str)

    skf = StratifiedKFold(
        n_splits=cfg["split"]["n_folds"], shuffle=True, random_state=SEED
    )
    dev_only["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(dev_only, dev_only["stratum"])):
        dev_only.loc[dev_only.index[val_idx], "fold"] = fold_idx

    # --- 3. Cross-hospital folds ---
    dev_only["fold_within_hospital"] = -1
    for hosp in ["A", "B"]:
        mask = dev_only["hospital"] == hosp
        hosp_df = dev_only[mask].reset_index()
        skf_h = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold_idx, (_, val_idx) in enumerate(
            skf_h.split(hosp_df, hosp_df["septic"])
        ):
            orig_indices = hosp_df.loc[val_idx, "index"].values
            dev_only.loc[orig_indices, "fold_within_hospital"] = fold_idx

    folds_df = dev_only[
        ["patient_id", "hospital", "septic", "fold", "fold_within_hospital"]
    ]
    folds_df.to_csv(folds_path, index=False)
    logger.info("Folds saved to %s: %d dev patients in %d folds", folds_path, len(folds_df), 5)

    return lockbox_df, folds_df


def get_rows(df: pd.DataFrame, patient_ids: pd.Series | set | list) -> pd.DataFrame:
    """Return rows of the given patients."""
    if not isinstance(patient_ids, set):
        patient_ids = set(patient_ids)
    return df[df["patient_id"].isin(patient_ids)]


def main() -> None:
    """CLI entrypoint for sepsis.splits."""
    parser = argparse.ArgumentParser(description="Create patient-level splits")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config path"
    )
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    create_splits(load_config(args.config))


if __name__ == "__main__":
    main()
