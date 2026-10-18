"""Physiological sanity cleaning and unit encoding for SepsisSentinel."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sepsis.config import load_config

logger = logging.getLogger(__name__)

# Physiological sanity ranges: inclusive [min, max]
SANITY_RANGES: Dict[str, Tuple[float, float]] = {
    "HR": (20.0, 300.0),
    "O2Sat": (50.0, 100.0),
    "Temp": (25.0, 45.0),
    "SBP": (40.0, 300.0),
    "MAP": (20.0, 250.0),
    "DBP": (10.0, 200.0),
    "Resp": (1.0, 80.0),
    "EtCO2": (1.0, 100.0),
    "pH": (6.5, 8.0),
    "Lactate": (0.1, 30.0),
    "WBC": (0.1, 200.0),
    "Platelets": (1.0, 2000.0),
    "Creatinine": (0.1, 25.0),
    "BUN": (1.0, 250.0),
    "Bilirubin_total": (0.1, 80.0),
    "Glucose": (10.0, 2000.0),
}


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw dataframe using physiological ranges and derive the unit column."""
    cleaned = df.copy()
    report_rows = []

    # 1. Apply standard physiological sanity ranges
    for var, (v_min, v_max) in SANITY_RANGES.items():
        if var not in cleaned.columns:
            continue
        vals = cleaned[var].values.copy()
        valid_mask = ~np.isnan(vals)
        n_observed = int(np.sum(valid_mask))

        out_of_bounds = valid_mask & ((vals < v_min) | (vals > v_max))
        n_invalid = int(np.sum(out_of_bounds))

        vals[out_of_bounds] = np.nan
        cleaned[var] = vals.astype(np.float32)

        pct_invalid = (n_invalid / n_observed * 100.0) if n_observed > 0 else 0.0
        report_rows.append(
            {
                "variable": var,
                "n_observed": n_observed,
                "n_set_to_nan": n_invalid,
                "pct_set_to_nan": pct_invalid,
            }
        )

    # 2. Special case FiO2
    if "FiO2" in cleaned.columns:
        fio2 = cleaned["FiO2"].values.copy()
        valid_mask = ~np.isnan(fio2)
        n_observed = int(np.sum(valid_mask))

        # Convert percentages > 1.0 and <= 100.0 to fractions
        pct_mask = valid_mask & (fio2 > 1.0) & (fio2 <= 100.0)
        fio2[pct_mask] = fio2[pct_mask] / 100.0

        # Keep only [0.21, 1.0], else NaN
        invalid_fio2 = valid_mask & ((fio2 < 0.21) | (fio2 > 1.0))
        n_invalid = int(np.sum(invalid_fio2))
        fio2[invalid_fio2] = np.nan
        cleaned["FiO2"] = fio2.astype(np.float32)

        pct_invalid = (n_invalid / n_observed * 100.0) if n_observed > 0 else 0.0
        report_rows.append(
            {
                "variable": "FiO2",
                "n_observed": n_observed,
                "n_set_to_nan": n_invalid,
                "pct_set_to_nan": pct_invalid,
            }
        )

    # 3. Unit encoding: 1 if Unit1==1, 2 if Unit2==1, 0 otherwise
    unit = np.zeros(len(cleaned), dtype=np.int8)
    if "Unit1" in cleaned.columns:
        unit[cleaned["Unit1"] == 1.0] = 1
    if "Unit2" in cleaned.columns:
        unit[cleaned["Unit2"] == 1.0] = 2
    cleaned["unit"] = unit

    report_df = pd.DataFrame(report_rows)
    return cleaned, report_df


def run_cleaning(cfg: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all_rows, apply cleaning, save clean_rows.parquet and cleaning report."""
    if cfg is None:
        cfg = load_config()

    in_path = Path(cfg["paths"]["all_rows"])
    out_path = Path(cfg["paths"]["clean_rows"])
    report_path = Path(cfg["paths"]["tables"]) / "cleaning_report.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s...", in_path)
    df = pd.read_parquet(in_path)

    logger.info("Cleaning data (shape: %s)...", df.shape)
    clean_df, report_df = clean(df)

    logger.info("Saving cleaning report to %s...", report_path)
    report_df.to_csv(report_path, index=False)

    logger.info("Saving cleaned rows to %s...", out_path)
    clean_df.to_parquet(out_path, engine="pyarrow", index=False)

    # Check acceptance: no variable loses more than 1% of its observed values
    print("\n--- P1.T3 Cleaning Report Summary ---")
    for _, row in report_df.iterrows():
        var = row["variable"]
        pct = row["pct_set_to_nan"]
        n_inv = int(row["n_set_to_nan"])
        n_obs = int(row["n_observed"])
        print(f"{var:16s}: {n_inv:6d} / {n_obs:8d} ({pct:6.3f}%) set to NaN")
        if pct > 1.0:
            logger.warning(
                "CRITICAL WARNING: %s lost %.2f%% of observed values (>1.0%%)!", var, pct
            )

    max_loss = report_df["pct_set_to_nan"].max()
    print(
        f"\nMaximum percentage lost by any variable: {max_loss:.3f}% (acceptance threshold: < 1.0%)"
    )
    print("------------------------------------\n")

    if max_loss > 1.0:
        raise ValueError(
            f"Acceptance failed: variable(s) lost >1% of observed values (max {max_loss:.2f}%)."
        )

    return clean_df, report_df


def main() -> None:
    """CLI entrypoint for sepsis.cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean all_rows.parquet and produce cleaning_report.csv"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    cfg = load_config(args.config)

    run_cleaning(cfg)


if __name__ == "__main__":
    main()
