"""Feature engineering for SepsisSentinel — P3.T1.

All operations are per patient, in hour order, using only current and past rows.
Uses pandas groupby().transform() for vectorised computation.
Total ≈ 298 features. Output: data/processed/features.parquet + models/feature_list.json.

Performance note: groupby().rolling() is used for slope computation — this is O(n·w) but
fully vectorised in C via pandas internals. Expect 5-15 minutes for 1.55M rows.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sepsis.config import load_config, set_seeds
from sepsis.constants import (
    BASELINE_VARS,
    LABS,
    LABS_ROLL,
    MEASURED,
    SEED,
    VITALS_ROLL,
)

logger = logging.getLogger(__name__)

CARRY_COLS = ["patient_id", "hospital", "hour", "SepsisLabel"]
FORBIDDEN_FEATURE_NAMES = {"hospital", "patient_id", "SepsisLabel", "Unit1", "Unit2"}


# ────────────────────────────────────────────────────────────────────
#  Vectorised rolling slope via groupby().rolling()
# ────────────────────────────────────────────────────────────────────


def _vectorized_rolling_slope(
    series: pd.Series,
    hours: pd.Series,
    pid: pd.Series,
    window: int,
    min_pts: int,
) -> pd.Series:
    """Compute per-patient rolling LS slope using groupby().transform().

    Computes: slope = (n·Sxy - Sx·Sy) / (n·Sxx - Sx²)
    where sums are over observed (non-NaN) values in the window.
    """
    m = series.notna().astype(float)
    y0 = series.fillna(0.0).astype(float)
    x = hours.astype(float)

    xm = x * m
    ym = y0 * m
    xym = x * m * y0
    xxm = x * x * m

    n_ = m.groupby(pid).transform(lambda s: s.rolling(window, min_periods=1).sum())
    Sx_ = xm.groupby(pid).transform(lambda s: s.rolling(window, min_periods=1).sum())
    Sy_ = ym.groupby(pid).transform(lambda s: s.rolling(window, min_periods=1).sum())
    Sxy_ = xym.groupby(pid).transform(lambda s: s.rolling(window, min_periods=1).sum())
    Sxx_ = xxm.groupby(pid).transform(lambda s: s.rolling(window, min_periods=1).sum())

    denom = n_ * Sxx_ - Sx_ * Sx_
    slope = (n_ * Sxy_ - Sx_ * Sy_) / denom
    slope = slope.where(n_ >= min_pts)
    slope = slope.where(denom.abs() >= 1e-12)
    return slope.astype("float32")


def _add_rolling_slope(
    df: pd.DataFrame, col: str, hour_col: str, window: int, min_pts: int, out_name: str
) -> pd.Series:
    """Legacy helper for backward compatibility."""
    return _vectorized_rolling_slope(df[col], df[hour_col], df["patient_id"], window, min_pts)


# ────────────────────────────────────────────────────────────────────
#  Single-patient helper for unit tests
# ────────────────────────────────────────────────────────────────────


def _rolling_slope_series(
    raw: pd.Series, mask: pd.Series, hours: pd.Series, window: int, min_points: int
) -> pd.Series:
    """Rolling LS slope for a single-patient series."""
    m = mask.astype(float)
    y0 = raw.fillna(0.0).astype(float)
    x = hours.astype(float)
    n = m.rolling(window, min_periods=1).sum()
    Sx = (x * m).rolling(window, min_periods=1).sum()
    Sy = (y0 * m).rolling(window, min_periods=1).sum()
    Sxy = (x * m * y0).rolling(window, min_periods=1).sum()
    Sxx = (x * x * m).rolling(window, min_periods=1).sum()
    denom = n * Sxx - Sx * Sx
    slope = (n * Sxy - Sx * Sy) / denom
    slope = slope.where(n >= min_points)
    slope = slope.where(denom.abs() >= 1e-12)
    return slope


def _apply_per_patient(df: pd.DataFrame, fn, include_groups: bool = False) -> pd.Series:
    """Apply fn per patient and always return a flat 1D Series aligned to df.index."""
    result = df.groupby("patient_id", sort=False).apply(fn, include_groups=include_groups)
    if isinstance(result, pd.DataFrame):
        flat = result.iloc[0]
        flat.index = df.index
        return flat
    else:
        return result.reset_index(level=0, drop=True).sort_index()


def _patient_features(g: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build all features for a single-patient df (used in unit tests)."""
    return build_features_from_df(g, cfg)


# ────────────────────────────────────────────────────────────────────
#  Vectorized feature builder
# ────────────────────────────────────────────────────────────────────


def build_features_from_df(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build all features from a (multi-patient) dataframe.

    df must have the same schema as clean_rows.parquet.
    Returns a dataframe with all feature columns + carry cols.
    """
    fcfg = cfg.get("features", {})
    tsl_cap = fcfg.get("tsl_cap_hours", 72)
    vital_windows = fcfg.get("vital_windows", [6, 12, 24])
    lab_windows = fcfg.get("lab_windows", [24])
    baseline_hours_cap = fcfg.get("baseline_hours", 6)

    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    pid = df["patient_id"]
    hours = df["hour"].astype(float)
    gr = df.groupby("patient_id", sort=False)

    cols: Dict[str, Any] = {}

    # ── F1: LOCF current values (34) ──
    for col in MEASURED:
        cols[col] = gr[col].transform(lambda s: s.ffill())

    # ── F2: Measured flag (34) ──
    for col in MEASURED:
        cols[f"{col}__meas"] = df[col].notna().astype("int8")

    # ── F3: Hours since last measurement (34) ──
    for col in MEASURED:
        last_obs = df["hour"].where(df[col].notna()).groupby(pid).transform(lambda s: s.ffill())
        tsl = hours - last_obs
        tsl = tsl.clip(upper=tsl_cap)
        tsl[last_obs.isna()] = tsl_cap
        cols[f"{col}__tsl"] = tsl.astype("float32")

    # ── F4: Lab count in last 24h (LABS) ──
    for col in LABS:
        cols[f"{col}__cnt24"] = (
            gr[col]
            .transform(lambda s: s.notna().astype(float).rolling(24, min_periods=1).sum())
            .astype("float32")
        )

    # ── F5: Rolling stats on OBSERVED values ──

    # VITALS_ROLL × {6,12,24} × {mean, min, max, std}
    for col in VITALS_ROLL:
        for w in vital_windows:
            cols[f"{col}__mean_{w}h"] = (
                gr[col].transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")
            )
            cols[f"{col}__min_{w}h"] = (
                gr[col].transform(lambda s: s.rolling(w, min_periods=1).min()).astype("float32")
            )
            cols[f"{col}__max_{w}h"] = (
                gr[col].transform(lambda s: s.rolling(w, min_periods=1).max()).astype("float32")
            )
            cols[f"{col}__std_{w}h"] = (
                gr[col].transform(lambda s: s.rolling(w, min_periods=2).std()).astype("float32")
            )

    # Slopes for VITALS_ROLL
    for col in VITALS_ROLL:
        for w in vital_windows:
            cols[f"{col}__slope_{w}h"] = _vectorized_rolling_slope(
                df[col], hours, pid, w, min_pts=3
            )

    # LABS_ROLL × {24} × {mean, min, max}
    for col in LABS_ROLL:
        for w in lab_windows:
            cols[f"{col}__mean_{w}h"] = (
                gr[col].transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")
            )
            cols[f"{col}__min_{w}h"] = (
                gr[col].transform(lambda s: s.rolling(w, min_periods=1).min()).astype("float32")
            )
            cols[f"{col}__max_{w}h"] = (
                gr[col].transform(lambda s: s.rolling(w, min_periods=1).max()).astype("float32")
            )

    # Slopes for LABS_ROLL
    for col in LABS_ROLL:
        for w in lab_windows:
            cols[f"{col}__slope_{w}h"] = _vectorized_rolling_slope(
                df[col], hours, pid, w, min_pts=2
            )

    # ── F6: 6-hour difference (VITALS_ROLL) ──
    for col in VITALS_ROLL:
        locf_vals = cols[col]
        cols[f"{col}__diff6"] = (locf_vals - locf_vals.groupby(pid).shift(6)).astype("float32")

    # ── F7: Baseline delta (BASELINE_VARS) ──
    within_bl = hours < baseline_hours_cap
    for col in BASELINE_VARS:
        raw = df[col]
        locf = cols[col]
        obs_masked = raw.fillna(0.0).where(within_bl, 0.0)
        count_bl = raw.notna().astype(float).where(within_bl, 0.0)
        cum_sum = obs_masked.groupby(pid).cumsum()
        cum_count = count_bl.groupby(pid).cumsum()
        baseline_early = cum_sum / cum_count.replace(0, np.nan)
        frozen = baseline_early.where(within_bl).groupby(pid).ffill()
        baseline_late = frozen.fillna(locf)
        baseline = baseline_early.where(within_bl, baseline_late)
        cols[f"{col}__dbase"] = (locf - baseline).astype("float32")

    # Forward fill raw unstandardized columns if present (for R2 per-hospital standardization)
    for col in MEASURED:
        raw_col = f"raw__{col}"
        if raw_col in df.columns:
            cols[raw_col] = df[raw_col].groupby(pid).ffill().astype("float32")

    # ── F8: Clinical indices ──
    hr_l = cols.get("raw__HR", cols["HR"])
    sbp_l = cols.get("raw__SBP", cols["SBP"])
    map_l = cols.get("raw__MAP", cols["MAP"])
    dbp_l = cols.get("raw__DBP", cols["DBP"])
    o2_l = cols.get("raw__O2Sat", cols["O2Sat"])
    fio2_l = cols.get("raw__FiO2", cols["FiO2"])
    bun_l = cols.get("raw__BUN", cols["BUN"])
    cr_l = cols.get("raw__Creatinine", cols["Creatinine"])

    with np.errstate(divide="ignore", invalid="ignore"):
        cols["idx__shock_index"] = (hr_l / sbp_l).astype("float32")
        cols["idx__mod_shock_index"] = (hr_l / map_l).astype("float32")
        cols["idx__pulse_pressure"] = (sbp_l - dbp_l).astype("float32")
        cols["idx__bun_cr"] = (bun_l / cr_l).astype("float32")
        cols["idx__sf_ratio"] = (o2_l / fio2_l).astype("float32")

    hr_raw = df["raw__HR"] if "raw__HR" in df.columns else df["HR"]
    sbp_raw = df["raw__SBP"] if "raw__SBP" in df.columns else df["SBP"]
    both_obs = hr_raw.notna() & sbp_raw.notna()
    si_raw = (hr_raw / sbp_raw).where(both_obs)
    cols["idx__shock_index__slope_6h"] = _vectorized_rolling_slope(si_raw, hours, pid, 6, min_pts=3)

    # ── F9: Clinical scores ──
    temp_l = cols.get("raw__Temp", cols["Temp"])
    resp_l = cols.get("raw__Resp", cols["Resp"])
    paco2_l = cols.get("raw__PaCO2", cols["PaCO2"])
    wbc_l = cols.get("raw__WBC", cols["WBC"])
    plt_l = cols.get("raw__Platelets", cols["Platelets"])
    bili_l = cols.get("raw__Bilirubin_total", cols["Bilirubin_total"])

    sirs = pd.Series(0, index=df.index, dtype="int8")
    sirs += ((temp_l > 38) | (temp_l < 36)).fillna(False).astype("int8")
    sirs += (hr_l > 90).fillna(False).astype("int8")
    sirs += ((resp_l > 20) | (paco2_l < 32)).fillna(False).astype("int8")
    sirs += ((wbc_l > 12) | (wbc_l < 4)).fillna(False).astype("int8")
    cols["score__sirs"] = sirs

    qsofa = pd.Series(0, index=df.index, dtype="int8")
    qsofa += (resp_l >= 22).fillna(False).astype("int8")
    qsofa += (sbp_l <= 100).fillna(False).astype("int8")
    cols["score__qsofa_partial"] = qsofa

    sofa = pd.Series(0.0, index=df.index, dtype="float32")
    sofa += np.where(
        plt_l < 20,
        4,
        np.where(plt_l < 50, 3, np.where(plt_l < 100, 2, np.where(plt_l < 150, 1, 0))),
    ).astype("float32") * plt_l.notna().astype("float32")
    sofa += np.where(
        bili_l >= 12,
        4,
        np.where(bili_l >= 6, 3, np.where(bili_l >= 2, 2, np.where(bili_l >= 1.2, 1, 0))),
    ).astype("float32") * bili_l.notna().astype("float32")
    sofa += np.where(
        cr_l >= 5, 4, np.where(cr_l >= 3.5, 3, np.where(cr_l >= 2, 2, np.where(cr_l >= 1.2, 1, 0)))
    ).astype("float32") * cr_l.notna().astype("float32")
    sofa += (map_l < 70).fillna(False).astype("float32")
    cols["score__sofa_partial"] = sofa

    sofa_min24 = sofa.groupby(pid).transform(lambda s: s.rolling(24, min_periods=1).min())
    cols["score__sofa_partial_d24"] = (sofa - sofa_min24).astype("float32")

    # ── F10: Static features ──
    cols["static__age"] = df["Age"].astype("float32")
    cols["static__gender"] = df["Gender"].astype("float32")
    cols["static__unit"] = df["unit"].astype("float32")
    cols["static__hosp_adm_time"] = df["HospAdmTime"].astype("float32")
    cols["static__iculos"] = (hours + 1).astype("float32")

    # ── Carry columns ──
    for c in CARRY_COLS:
        cols[c] = df[c].values

    return pd.DataFrame(cols, index=df.index)


# ────────────────────────────────────────────────────────────────────
#  Main build entry (chunk-based with multiprocessing)
# ────────────────────────────────────────────────────────────────────


def _chunk_worker(args):
    chunk_df, cfg = args
    return build_features_from_df(chunk_df, cfg)


def build_features(
    clean_df: Optional[pd.DataFrame] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Build the full feature matrix. Processes in chunks with multiprocessing."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    if cfg is None:
        cfg = load_config()
    set_seeds(cfg.get("seed", SEED))

    if clean_df is None:
        clean_path = Path(cfg["paths"]["clean_rows"])
        logger.info("Loading clean_rows from %s...", clean_path)
        clean_df = pd.read_parquet(clean_path)

    out_path = Path(cfg["paths"]["features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    pids = clean_df["patient_id"].unique()
    total = len(pids)
    # Target ~16 chunks for optimal parallel throughput
    n_workers = min(8, os.cpu_count() or 4)
    chunk_size = max(500, int(np.ceil(total / (n_workers * 2))))
    n_chunks = (total + chunk_size - 1) // chunk_size
    logger.info(
        "Building features: %d rows, %d patients in %d chunks (using %d workers)",
        len(clean_df),
        total,
        n_chunks,
        n_workers,
    )

    chunk_tasks = []
    for i in range(n_chunks):
        chunk_pids = pids[i * chunk_size : (i + 1) * chunk_size]
        chunk_df = clean_df[clean_df["patient_id"].isin(chunk_pids)].copy()
        chunk_tasks.append((chunk_df, cfg))

    if n_workers > 1 and len(chunk_tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            frames = list(executor.map(_chunk_worker, chunk_tasks))
    else:
        frames = [_chunk_worker(t) for t in chunk_tasks]

    features_df = pd.concat(frames, ignore_index=True)
    logger.info("Feature matrix: %s", features_df.shape)

    feature_cols = [
        c
        for c in features_df.columns
        if c not in CARRY_COLS and c not in FORBIDDEN_FEATURE_NAMES and not c.startswith("raw__")
    ]
    raw_cols_to_drop = [c for c in features_df.columns if c.startswith("raw__")]
    if raw_cols_to_drop:
        features_df = features_df.drop(columns=raw_cols_to_drop)

    feature_list_path = models_dir / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info("Feature list (%d features) saved to %s", len(feature_cols), feature_list_path)

    features_df.to_parquet(out_path, index=False)
    logger.info("Features saved to %s", out_path)
    return features_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature matrix")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    build_features(cfg=load_config(args.config))


if __name__ == "__main__":
    main()
