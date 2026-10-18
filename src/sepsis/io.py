"""Data loading and raw-table construction for SepsisSentinel."""

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sepsis.config import load_config
from sepsis.constants import MEASURED, RAW_COLUMNS, STATIC

logger = logging.getLogger(__name__)


def read_patient(path: Path | str) -> pd.DataFrame:
    """Read a single patient PSV file and validate its 41 columns."""
    p = Path(path)
    df = pd.read_csv(p, sep="|")
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(
            f"Columns in {p} do not match expected RAW_COLUMNS order or names.\n"
            f"Expected: {RAW_COLUMNS}\nFound: {list(df.columns)}"
        )
    return df


def _process_file_batch(file_batch: List[Tuple[Path, str]]) -> pd.DataFrame:
    """Process a batch of patient files into a single concatenated DataFrame."""
    dfs: List[pd.DataFrame] = []
    for path, hospital in file_batch:
        df = pd.read_csv(path, sep="|")
        n_rows = len(df)
        df["patient_id"] = path.stem
        df["hospital"] = hospital
        df["hour"] = np.arange(n_rows, dtype=np.int16)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_all_rows(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Build all_rows table from raw PSV files and save to Parquet."""
    if cfg is None:
        cfg = load_config()

    raw_a_dir = Path(cfg["paths"]["raw_a"])
    raw_b_dir = Path(cfg["paths"]["raw_b"])
    out_path = Path(cfg["paths"]["all_rows"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files_a = sorted(raw_a_dir.glob("*.psv"))
    files_b = sorted(raw_b_dir.glob("*.psv"))

    logger.info("Found %d files in Set A, %d files in Set B", len(files_a), len(files_b))

    items: List[Tuple[Path, str]] = [(f, "A") for f in files_a] + [(f, "B") for f in files_b]

    # Batch files to optimize ProcessPoolExecutor IPC overhead
    batch_size = 500
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    logger.info(
        "Processing %d files in %d batches using %d workers",
        len(items),
        len(batches),
        max_workers,
    )
    t0 = time.time()

    dfs: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_df in tqdm(
            executor.map(_process_file_batch, batches),
            total=len(batches),
            desc="Loading PSVs",
        ):
            dfs.append(batch_df)

    logger.info("Concatenating all batch dataframes...")
    full_df = pd.concat(dfs, ignore_index=True)

    # Cast types: measurements, static, and label to float32
    float_cols = MEASURED + STATIC + ["SepsisLabel"]
    for col in float_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(np.float32)

    # Ensure patient_id is string and hour is int16
    full_df["patient_id"] = full_df["patient_id"].astype(str)
    full_df["hospital"] = full_df["hospital"].astype(str)
    full_df["hour"] = full_df["hour"].astype(np.int16)

    logger.info("Sorting by patient_id, hour...")
    full_df = full_df.sort_values(["patient_id", "hour"]).reset_index(drop=True)

    logger.info("Writing all_rows to %s...", out_path)
    full_df.to_parquet(out_path, engine="pyarrow", index=False)
    elapsed = time.time() - t0
    logger.info("Saved %s in %.2f seconds. Shape: %s", out_path, elapsed, full_df.shape)

    return full_df


def verify_all_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute and print verification stats according to P1.T2 acceptance checks."""
    n_unique_patients = df["patient_id"].nunique()
    patients_by_hosp = df.groupby("hospital")["patient_id"].nunique().to_dict()
    total_rows = len(df)

    patient_septic = df.groupby("patient_id")["SepsisLabel"].max()
    n_septic_patients = int((patient_septic == 1.0).sum())
    septic_patient_pct = (n_septic_patients / n_unique_patients) * 100.0

    pos_rows = (df["SepsisLabel"] == 1.0).sum()
    pos_row_pct = (pos_rows / total_rows) * 100.0

    # Check ICULOS step
    df_sorted = df.sort_values(["patient_id", "hour"])
    iculos_diff = df_sorted.groupby("patient_id")["ICULOS"].diff()
    # For rows after the first row (diff is not NaN), check if diff != 1.0
    non_first_mask = iculos_diff.notna()
    invalid_iculos_mask = non_first_mask & (iculos_diff != 1.0)
    patients_with_invalid_iculos = df_sorted.loc[invalid_iculos_mask, "patient_id"].nunique()

    stats = {
        "unique_patients": n_unique_patients,
        "patients_by_hospital": patients_by_hosp,
        "total_rows": total_rows,
        "septic_patients": n_septic_patients,
        "septic_patient_pct": septic_patient_pct,
        "positive_rows": int(pos_rows),
        "positive_row_pct": pos_row_pct,
        "patients_with_invalid_iculos_step": patients_with_invalid_iculos,
    }

    hosp_a_pts = patients_by_hosp.get("A", 0)
    hosp_b_pts = patients_by_hosp.get("B", 0)

    print("\n--- P1.T2 Acceptance Verification ---")
    print(f"1. Unique patients: {n_unique_patients} (expected 40,336)")
    print(
        f"2. Patients per hospital: A = {hosp_a_pts}, B = {hosp_b_pts} (expected 20,336 / 20,000)"
    )
    print(f"3. Total rows: {total_rows} (expected ~1.55M)")
    print(
        f"4. Septic patients: {n_septic_patients} ({septic_patient_pct:.2f}%) "
        f"(expected ~2,932 / ~7.3%)"
    )
    print(f"5. Share of positive rows: {pos_row_pct:.2f}% (expected ~2%)")
    print(
        f"6. Patients where ICULOS does not increase by 1 per row: {patients_with_invalid_iculos}"
    )
    print("-------------------------------------\n")

    return stats


def main() -> None:
    """CLI entrypoint for sepsis.io."""
    parser = argparse.ArgumentParser(description="Build and verify all_rows.parquet from raw PSVs")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing all_rows.parquet"
    )
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    cfg = load_config(args.config)

    if args.verify_only:
        out_path = Path(cfg["paths"]["all_rows"])
        logger.info("Reading %s for verification...", out_path)
        df = pd.read_parquet(out_path)
    else:
        df = build_all_rows(cfg)

    verify_all_rows(df)


if __name__ == "__main__":
    main()
