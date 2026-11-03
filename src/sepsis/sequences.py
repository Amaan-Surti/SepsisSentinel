"""Per-patient hourly sequences for the GRU comparison — P6.T1.

Dev patients only: lockbox rows are filtered out at read time, so this module
never loads them. Input layout per hour (111 dims, float32):
    34 LOCF values | 34 measured flags | 34 hours-since-last / 72 |
    SIRS / 4 | partial SOFA / 13 | Age / 100 | Gender | unit one-hot (3) |
    clip(HospAdmTime, -1000, 0) / 1000 | (hour + 1) / 336

LOCF values stay unscaled on disk. `standardize` z-scores them per fold, using
observed values from that fold's training patients only.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sepsis.config import load_config
from sepsis.constants import MEASURED

logger = logging.getLogger(__name__)

N_MEASURED = len(MEASURED)
MEAS_START = N_MEASURED
TSL_START = 2 * N_MEASURED
TSL_CAP = 72.0
MAX_HOURS = 336.0


def input_columns() -> list:
    """Feature columns read to build a sequence row."""
    return (
        list(MEASURED)
        + [f"{c}__meas" for c in MEASURED]
        + [f"{c}__tsl" for c in MEASURED]
        + [
            "score__sirs",
            "score__sofa_partial",
            "static__age",
            "static__gender",
            "static__unit",
            "static__hosp_adm_time",
        ]
    )


def rows_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Turn feature rows (sorted by patient, hour) into the unscaled input matrix."""
    n = len(df)
    tsl = df[[f"{c}__tsl" for c in MEASURED]].to_numpy("float32") / TSL_CAP
    unit = df["static__unit"].fillna(0).astype(int).to_numpy()
    extras = np.column_stack(
        [
            df["score__sirs"].to_numpy("float32") / 4.0,
            df["score__sofa_partial"].to_numpy("float32") / 13.0,
            np.nan_to_num(df["static__age"].to_numpy("float32")) / 100.0,
            np.nan_to_num(df["static__gender"].to_numpy("float32")),
            unit == 0,
            unit == 1,
            unit == 2,
            np.clip(np.nan_to_num(df["static__hosp_adm_time"].to_numpy("float32")), -1000, 0)
            / 1000.0,
            (df["hour"].to_numpy("float32") + 1.0) / MAX_HOURS,
        ]
    )
    X = np.empty((n, TSL_START + N_MEASURED + extras.shape[1]), dtype="float32")
    X[:, :N_MEASURED] = df[list(MEASURED)].to_numpy("float32")
    X[:, MEAS_START:TSL_START] = df[[f"{c}__meas" for c in MEASURED]].to_numpy("float32")
    X[:, TSL_START : TSL_START + N_MEASURED] = tsl
    X[:, TSL_START + N_MEASURED :] = extras
    return X


def standardize(X: np.ndarray, train_rows: np.ndarray) -> Tuple[np.ndarray, Dict[str, list]]:
    """Z-score the LOCF block with stats from observed training values; NaN becomes 0."""
    observed = X[train_rows, MEAS_START:TSL_START] > 0.5
    values = np.where(observed, X[train_rows, :N_MEASURED], np.nan)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 0, std, 1.0)
    norm = {"mean": mean.tolist(), "std": std.tolist()}
    return apply_norm(X, norm), norm


def apply_norm(X: np.ndarray, norm: Dict[str, list]) -> np.ndarray:
    """Apply saved LOCF z-score stats (float32, as in training); NaN becomes 0."""
    mean = np.asarray(norm["mean"], dtype="float32")
    std = np.asarray(norm["std"], dtype="float32")
    out = X.copy()
    out[:, :N_MEASURED] = np.nan_to_num((X[:, :N_MEASURED] - mean) / std).astype("float32")
    return out


def standardize_extra(
    E: np.ndarray, train_rows: np.ndarray
) -> Tuple[np.ndarray, Dict[str, list]]:
    """Z-score appended LightGBM feature columns with training-row stats; NaN becomes 0."""
    mean = np.nanmean(E[train_rows], axis=0)
    std = np.nanstd(E[train_rows], axis=0)
    mean = np.nan_to_num(mean)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    norm = {"mean": mean.tolist(), "std": std.tolist()}
    return apply_extra_norm(E, norm), norm


def apply_extra_norm(E: np.ndarray, norm: Dict[str, list]) -> np.ndarray:
    """Apply saved stats for the appended feature columns."""
    mean = np.asarray(norm["mean"], dtype="float32")
    std = np.asarray(norm["std"], dtype="float32")
    return np.nan_to_num((E.astype("float32") - mean) / std).astype("float32")


def build_sequences(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Write X.npy, y.npy, hours.npy, offsets.npy and patients.csv for dev patients."""
    if cfg is None:
        cfg = load_config()
    paths = cfg["paths"]
    folds = pd.read_csv(paths["folds"])
    splits = pd.read_csv(paths["lockbox"])
    lockbox = set(splits.loc[splits["split"] == "test", "patient_id"])
    if lockbox & set(folds["patient_id"]):
        raise ValueError("Fold file overlaps the lockbox")

    cols = ["patient_id", "hour", "SepsisLabel"] + input_columns()
    df = pd.read_parquet(
        paths["features"],
        columns=cols,
        filters=[("patient_id", "in", folds["patient_id"].tolist())],
    )
    if lockbox & set(df["patient_id"].unique()):
        raise ValueError("Lockbox rows reached the sequence builder")
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)

    pids, starts = np.unique(df["patient_id"].to_numpy(), return_index=True)
    offsets = np.r_[starts, len(df)].astype("int64")
    meta = pd.DataFrame({"patient_id": pids}).merge(
        folds[["patient_id", "hospital", "septic", "fold"]], on="patient_id", validate="1:1"
    )

    out_dir = Path(paths["features"]).parent / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", rows_to_matrix(df))
    np.save(out_dir / "y.npy", df["SepsisLabel"].to_numpy("int8"))
    np.save(out_dir / "hours.npy", df["hour"].to_numpy("int32"))
    np.save(out_dir / "offsets.npy", offsets)
    meta.to_csv(out_dir / "patients.csv", index=False)
    with open(out_dir / "columns.json", "w") as f:
        json.dump({"input_columns": input_columns(), "n_rows": len(df)}, f, indent=2)
    logger.info("Wrote %d rows for %d dev patients to %s", len(df), len(meta), out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dev-only GRU sequences")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_sequences()


if __name__ == "__main__":
    main()
