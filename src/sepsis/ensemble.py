"""LightGBM vs GRU vs ensemble on dev OOF predictions — P6.T3.

Dev-only comparison; the lockbox is not read. Two scores are reported:
    pooled: calibrate and search the policy on all OOF rows, then score those rows.
            This is how the LightGBM dev utility of 0.4267 was produced.
    nested: for each fold, calibration, blend weight and policy come from the other
            folds only. This is the fairer estimate, since nothing is fit on scored rows.
The paired bootstrap resamples patients and compares nested alerts between models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from sepsis.alerting import apply_policy, search_policy
from sepsis.config import load_config
from sepsis.constants import SEED
from sepsis.metrics import evaluate_predictions
from sepsis.utility import _best_predictions_for_patient, compute_prediction_utility

logger = logging.getLogger(__name__)

# P6 grid (gru_dev_comparison*.csv); 0.7 was on its edge, so P9 widened it once to WEIGHTS
P6_WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7)
WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9)
METRICS = [
    "utility_norm",
    "auroc",
    "auprc",
    "ece",
    "patient_sensitivity",
    "median_lead_time_h",
    "false_alerts_per_patient_day",
]


def load_oof(gru_path: str = "models/gru/oof.parquet") -> pd.DataFrame:
    """Join LightGBM and GRU OOF predictions and attach folds."""
    cfg = load_config()
    lgbm = pd.read_parquet("models/lgbm_tuned/oof.parquet").rename(columns={"prob": "lgbm"})
    gru = pd.read_parquet(gru_path).rename(columns={"prob": "gru"})
    df = lgbm.merge(
        gru[["patient_id", "hour", "gru"]], on=["patient_id", "hour"], validate="1:1", how="inner"
    )
    if not len(df) == len(lgbm) == len(gru):
        raise ValueError("LightGBM and GRU OOF rows do not align")
    folds = pd.read_csv(cfg["paths"]["folds"])[["patient_id", "fold"]]
    df = df.merge(folds, on="patient_id", validate="m:1")
    return df.sort_values(["patient_id", "hour"]).reset_index(drop=True)


def _fit_iso(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(probs, labels)


def _policy(df: pd.DataFrame, col: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return search_policy(df[["patient_id", "hour", "SepsisLabel", col]], col, cfg)[0]


def fit_score(
    train: pd.DataFrame,
    score: pd.DataFrame,
    model: str,
    cfg: Dict[str, Any],
    weights: Sequence[float] = P6_WEIGHTS,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Fit calibration, blend weight and policy on train; return calibrated score probs."""
    labels = train["SepsisLabel"].to_numpy(int)
    params: Dict[str, Any] = {}
    if model in ("lgbm", "gru"):
        iso = _fit_iso(train[model].to_numpy(), labels)
        train_cal = iso.predict(train[model].to_numpy())
        score_cal = iso.predict(score[model].to_numpy())
    else:
        iso_l = _fit_iso(train["lgbm"].to_numpy(), labels)
        iso_g = _fit_iso(train["gru"].to_numpy(), labels)
        tl, tg = iso_l.predict(train["lgbm"].to_numpy()), iso_g.predict(train["gru"].to_numpy())
        best_u = -np.inf
        for w in weights:
            blend = train[["patient_id", "hour", "SepsisLabel"]].assign(p=w * tl + (1 - w) * tg)
            p = _policy(blend, "p", cfg)
            if p["utility_norm"] > best_u:
                best_u, params["weight"] = p["utility_norm"], w
        w = params["weight"]
        train_cal = w * tl + (1 - w) * tg
        score_cal = w * iso_l.predict(score["lgbm"].to_numpy()) + (1 - w) * iso_g.predict(
            score["gru"].to_numpy()
        )
    train_out = train[["patient_id", "hour", "SepsisLabel"]].assign(p=train_cal)
    params.update(_policy(train_out, "p", cfg))
    return pd.Series(score_cal, index=score.index), params


def evaluate_model(df: pd.DataFrame, model: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pooled and nested results for one model; adds a nested alert column to df."""
    probs, params = fit_score(df, df, model, cfg)
    pooled = df[["patient_id", "hour", "SepsisLabel"]].assign(p=probs)
    pooled["a"] = apply_policy(pooled, "p", params["threshold"], params["k"], params["sticky"])
    pooled_m = evaluate_predictions(pooled, "p", "a", cfg)

    nested = df[["patient_id", "hour", "SepsisLabel"]].assign(p=np.nan, a=0)
    fold_params = []
    for fold in sorted(df["fold"].unique()):
        tr, te = df[df["fold"] != fold], df[df["fold"] == fold]
        te_p, p = fit_score(tr, te, model, cfg)
        scored = te[["patient_id", "hour", "SepsisLabel"]].assign(p=te_p)
        nested.loc[te.index, "p"] = te_p
        nested.loc[te.index, "a"] = np.asarray(
            apply_policy(scored, "p", p["threshold"], p["k"], p["sticky"])
        ).astype(int)
        fold_params.append({"fold": int(fold), **p})
    nested_m = evaluate_predictions(nested, "p", "a", cfg)
    logger.info("%s pooled %.4f nested %.4f", model, pooled_m["utility_norm"],
                nested_m["utility_norm"])
    return {
        "pooled": pooled_m,
        "nested": nested_m,
        "params": params,
        "fold_params": fold_params,
        "alerts": nested["a"].to_numpy(int),
    }


def patient_utility_arrays(
    df: pd.DataFrame, alerts: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-patient observed, best-possible and do-nothing utility."""
    u = cfg["utility"]
    args = (
        u["dt_early"], u["dt_optimal"], u["dt_late"],
        u["max_u_tp"], u["min_u_fn"], u["u_fp"], u["u_tn"],
    )
    labels = df["SepsisLabel"].to_numpy(int)
    pids = df["patient_id"].to_numpy()
    edges = np.r_[0, np.where(pids[:-1] != pids[1:])[0] + 1, len(df)]
    obs, best, zero = [], [], []
    for s, e in zip(edges[:-1], edges[1:]):
        lab = labels[s:e]
        best_p = _best_predictions_for_patient(lab, u["dt_early"], u["dt_optimal"], u["dt_late"])
        obs.append(compute_prediction_utility(lab, alerts[s:e], *args))
        best.append(compute_prediction_utility(lab, best_p, *args))
        zero.append(compute_prediction_utility(lab, np.zeros(e - s, dtype=int), *args))
    return np.array(obs), np.array(best), np.array(zero)


def paired_bootstrap(
    df: pd.DataFrame,
    alerts_a: np.ndarray,
    alerts_b: np.ndarray,
    cfg: Dict[str, Any],
    n: int = 1000,
) -> Dict[str, float]:
    """Bootstrap the utility difference a - b over patients."""
    obs_a, best, zero = patient_utility_arrays(df, alerts_a, cfg)
    obs_b, _, _ = patient_utility_arrays(df, alerts_b, cfg)

    def norm(o: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (o.sum(-1) - z.sum(-1)) / (b.sum(-1) - z.sum(-1))

    rng = np.random.default_rng(SEED)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(best), len(best))
        diffs[i] = norm(obs_a[idx], best[idx], zero[idx]) - norm(obs_b[idx], best[idx], zero[idx])
    return {
        "diff": float(norm(obs_a, best, zero) - norm(obs_b, best, zero)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_better": float((diffs > 0).mean()),
    }


def run_comparison(
    cfg: Optional[Dict[str, Any]] = None,
    gru_path: str = "models/gru/oof.parquet",
    out_name: str = "gru_dev_comparison",
) -> pd.DataFrame:
    """Write the dev comparison table and paired bootstrap results."""
    if cfg is None:
        cfg = load_config()
    df = load_oof(gru_path)
    results = {m: evaluate_model(df, m, cfg) for m in ("lgbm", "gru", "ensemble")}

    rows = []
    for name, r in results.items():
        for scheme in ("pooled", "nested"):
            row = {"model": name, "scheme": scheme}
            row.update({k: r[scheme][k] for k in METRICS})
            row.update({k: r["params"].get(k) for k in ("threshold", "k", "sticky", "weight")})
            rows.append(row)
    table = pd.DataFrame(rows)
    tables = Path(cfg["paths"]["tables"])
    table.to_csv(tables / f"{out_name}.csv", index=False)

    boot = pd.DataFrame(
        [
            {
                "comparison": f"{m} - lgbm",
                **paired_bootstrap(df, results[m]["alerts"], results["lgbm"]["alerts"], cfg),
            }
            for m in ("gru", "ensemble")
        ]
    )
    boot.to_csv(tables / f"{out_name}_bootstrap.csv", index=False)
    pd.DataFrame(
        [{"model": m, **p} for m, r in results.items() for p in r["fold_params"]]
    ).to_csv(tables / f"{out_name}_fold_params.csv", index=False)
    print(table.to_string())
    print(boot.to_string())
    return table


# ------------------------------------------------------------------ P9: widened nested search

ID_COLS = ["patient_id", "hour", "SepsisLabel"]
GRU_VARIANTS = {
    "plain": "models/gru/oof.parquet",
    "h64": "models/gru_h64/oof.parquet",
    "top32": "models/gru_top32/oof.parquet",
}
JobKey = Tuple[str, Optional[int], str, Optional[float]]


def calibrated_probs(
    train: pd.DataFrame, score: pd.DataFrame, model: str, weight: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotonic fit on train; (train, score) probabilities for one model or a fixed blend."""
    labels = train["SepsisLabel"].to_numpy(int)

    def cal(col: str) -> Tuple[np.ndarray, np.ndarray]:
        iso = _fit_iso(train[col].to_numpy(), labels)
        return iso.predict(train[col].to_numpy()), iso.predict(score[col].to_numpy())

    if model in ("lgbm", "gru"):
        return cal(model)
    if weight is None:
        raise ValueError("The ensemble needs a weight")
    (tl, sl), (tg, sg) = cal("lgbm"), cal("gru")
    return weight * tl + (1 - weight) * tg, weight * sl + (1 - weight) * sg


def fit_job(
    train: pd.DataFrame,
    score: pd.DataFrame,
    model: str,
    weight: Optional[float],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Calibrate (and blend at a fixed weight) on train, search the policy on train, score."""
    train_p, score_p = calibrated_probs(train, score, model, weight)
    policy = _policy(train[ID_COLS].assign(p=train_p), "p", cfg)
    scored = score[ID_COLS].assign(p=score_p)
    alerts = apply_policy(scored, "p", policy["threshold"], policy["k"], policy["sticky"])
    return {"policy": policy, "probs": score_p, "alerts": np.asarray(alerts).astype(int)}


def run_jobs(
    frames: Dict[str, pd.DataFrame], keys: List[JobKey], cfg: Dict[str, Any], n_jobs: int
) -> Dict[JobKey, Dict[str, Any]]:
    """Run fit_job for (variant, fold or None = pooled, model, weight) keys in parallel."""
    from joblib import Parallel, delayed

    cols = ID_COLS + ["lgbm", "gru"]

    def split(name: str, fold: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = frames[name]
        if fold is None:
            return df[cols], df[cols]
        held = df["fold"].to_numpy() == fold
        return df.loc[~held, cols], df.loc[held, cols]

    tasks = (delayed(fit_job)(*split(v, f), m, w, cfg) for v, f, m, w in keys)
    return dict(zip(keys, Parallel(n_jobs=n_jobs, verbose=5)(tasks)))


def assemble(
    df: pd.DataFrame,
    jobs: Dict[JobKey, Dict[str, Any]],
    variant: str,
    model: str,
    cfg: Dict[str, Any],
    weight: Optional[float] = None,
) -> Dict[str, Any]:
    """Pooled and nested results. For the ensemble with weight=None, each training set
    picks its own weight (highest in-sample policy utility; ties go to the lower weight)."""

    def pick(fold: Optional[int]) -> Optional[float]:
        if model != "ensemble" or weight is not None:
            return weight
        best_w, best_u = None, -np.inf
        for w in WEIGHTS:
            u = jobs[(variant, fold, model, w)]["policy"]["utility_norm"]
            if u > best_u:
                best_w, best_u = w, u
        return best_w

    pooled_w = pick(None)
    pj = jobs[(variant, None, model, pooled_w)]
    pooled = df[ID_COLS].assign(p=pj["probs"], a=pj["alerts"])
    nested = df[ID_COLS].assign(p=np.nan, a=0)
    fold_arr = df["fold"].to_numpy()
    fold_params = []
    for fold in sorted(int(f) for f in np.unique(fold_arr)):
        w = pick(fold)
        j = jobs[(variant, fold, model, w)]
        held = fold_arr == fold
        nested.loc[held, "p"] = j["probs"]
        nested.loc[held, "a"] = j["alerts"]
        fold_params.append({"fold": fold, **j["policy"], "weight": w})
    return {
        "pooled": evaluate_predictions(pooled, "p", "a", cfg),
        "nested": evaluate_predictions(nested, "p", "a", cfg),
        "params": {**pj["policy"], "weight": pooled_w},
        "fold_params": fold_params,
        "alerts": nested["a"].to_numpy(int),
    }


def run_selection(
    cfg: Optional[Dict[str, Any]] = None,
    variants: Optional[Dict[str, str]] = None,
    n_jobs: int = 16,
    out_name: str = "ensemble_dev",
) -> Dict[str, Any]:
    """Widened weight search and GRU-variant check, all nested on dev OOF; writes tables."""
    if cfg is None:
        cfg = load_config()
    variants = variants or GRU_VARIANTS
    frames = {name: load_oof(path) for name, path in variants.items()}
    plain = frames["plain"]
    for name, df in frames.items():
        if not np.array_equal(df["patient_id"].to_numpy(), plain["patient_id"].to_numpy()):
            raise ValueError(f"{name} OOF rows are not in the same order as the plain GRU")
    folds: List[Optional[int]] = [None, *sorted(int(f) for f in plain["fold"].unique())]

    keys: List[JobKey] = [("plain", f, "lgbm", None) for f in folds]
    for name in frames:
        keys += [(name, f, "gru", None) for f in folds]
        keys += [(name, f, "ensemble", w) for f in folds for w in WEIGHTS]
    jobs = run_jobs(frames, keys, cfg, n_jobs)

    results = {("lgbm", "-"): assemble(plain, jobs, "plain", "lgbm", cfg)}
    weight_rows = []
    for name, df in frames.items():
        results[("gru", name)] = assemble(df, jobs, name, "gru", cfg)
        ens = results[("ensemble", name)] = assemble(df, jobs, name, "ensemble", cfg)
        for w in WEIGHTS:
            r = assemble(df, jobs, name, "ensemble", cfg, weight=w)
            weight_rows.append(
                {
                    "gru_variant": name,
                    "weight_lgbm": w,
                    **{f"nested_{k}": r["nested"][k] for k in METRICS},
                    "nested_policy_threshold": ",".join(
                        f"{p['threshold']:.2f}" for p in r["fold_params"]
                    ),
                    "pooled_utility_norm": r["pooled"]["utility_norm"],
                    "folds_selecting": sum(p["weight"] == w for p in ens["fold_params"]),
                    "pooled_selects": ens["params"]["weight"] == w,
                }
            )

    rows = []
    for (model, name), r in results.items():
        for scheme in ("pooled", "nested"):
            row = {"model": model, "gru_variant": name, "scheme": scheme}
            row.update({k: r[scheme][k] for k in METRICS})
            row.update({k: r["params"].get(k) for k in ("threshold", "k", "sticky", "weight")})
            rows.append(row)

    def boot(a: Tuple[str, str], b: Tuple[str, str]) -> Dict[str, Any]:
        res = paired_bootstrap(plain, results[a]["alerts"], results[b]["alerts"], cfg)
        return {"comparison": f"{'/'.join(a)} - {'/'.join(b)}", **res}

    lgbm = ("lgbm", "-")
    boots = [boot(("gru", "plain"), lgbm), boot(("ensemble", "plain"), lgbm)]
    for name in frames:
        if name != "plain":
            boots.append(boot(("ensemble", name), lgbm))
            boots.append(boot(("ensemble", name), ("ensemble", "plain")))

    others = [n for n in frames if n != "plain"]
    chosen, reason = "plain", "no variant was run"
    if others:
        best = max(others, key=lambda n: results[("ensemble", n)]["nested"]["utility_norm"])
        best_u = results[("ensemble", best)]["nested"]["utility_norm"]
        plain_u = results[("ensemble", "plain")]["nested"]["utility_norm"]
        ci_low = next(
            b["ci_low"] for b in boots if b["comparison"] == f"ensemble/{best} - ensemble/plain"
        )
        if best_u > plain_u and ci_low > 0:
            chosen, reason = best, f"best nested ensemble utility and CI low {ci_low:+.4f} > 0"
        else:
            reason = (
                f"best variant {best} nested {best_u:.4f} vs plain {plain_u:.4f}; "
                f"bootstrap CI low {ci_low:+.4f} is not above 0, so the plain GRU is kept"
            )

    tables = Path(cfg["paths"]["tables"])
    pd.DataFrame(rows).to_csv(tables / f"{out_name}_comparison.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(tables / f"{out_name}_weights.csv", index=False)
    pd.DataFrame(boots).to_csv(tables / f"{out_name}_bootstrap.csv", index=False)
    pd.DataFrame(
        [
            {"model": m, "gru_variant": n, **p}
            for (m, n), r in results.items()
            for p in r["fold_params"]
        ]
    ).to_csv(tables / f"{out_name}_fold_params.csv", index=False)
    chosen_ens = results[("ensemble", chosen)]
    selection = {
        "weights": list(WEIGHTS),
        "variants": variants,
        "chosen_variant": chosen,
        "reason": reason,
        "pooled_weight": chosen_ens["params"]["weight"],
        "pooled_policy": {k: chosen_ens["params"][k] for k in ("threshold", "k", "sticky")},
        "nested_utility": chosen_ens["nested"]["utility_norm"],
    }
    with open(tables / f"{out_name}_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(weight_rows).to_string())
    print(pd.DataFrame(boots).to_string())
    print(json.dumps(selection, indent=2))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LightGBM, GRU and ensemble on dev OOF")
    parser.add_argument("--gru-path", default="models/gru/oof.parquet")
    parser.add_argument("--out-name", default=None)
    parser.add_argument(
        "--select", action="store_true", help="P9 widened weight search and GRU-variant check"
    )
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.select:
        run_selection(n_jobs=args.n_jobs, out_name=args.out_name or "ensemble_dev")
    else:
        run_comparison(gru_path=args.gru_path, out_name=args.out_name or "gru_dev_comparison")


if __name__ == "__main__":
    main()
