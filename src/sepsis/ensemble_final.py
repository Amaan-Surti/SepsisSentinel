"""Final LightGBM + GRU ensemble, fit on development data only — P9.

1. GRU: the variant chosen by `ensemble --select`, trained on all 32,268 dev patients for
   median(best fold epoch) + 1 epochs, focal alpha and config from that variant's meta.json.
2. Calibration: LightGBM keeps the frozen models/final/calibrator.joblib; the GRU gets an
   isotonic fit on its dev OOF predictions.
3. Blend weight: the one the pooled dev OOF selects under the nested rule (ensemble_dev_selection).
4. Policy: search_policy on the blended calibrated dev OOF.

The lockbox is never read. models/final/ is only read.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from sepsis.alerting import apply_policy, search_policy
from sepsis.config import load_config
from sepsis.ensemble import ID_COLS, METRICS, WEIGHTS, _fit_iso, load_oof
from sepsis.metrics import evaluate_predictions
from sepsis.train_gru import git_sha, top_lgbm_features, train_final

logger = logging.getLogger(__name__)


def final_epochs(meta: Dict[str, Any]) -> int:
    """Median best epoch (0-indexed) across folds, + 1 = number of epochs to train."""
    return int(np.median([f["best_epoch"] for f in meta["folds"]])) + 1


def build_final_ensemble(
    cfg: Optional[Dict[str, Any]] = None,
    out_dir: str | Path = "models/ensemble_final",
    lgbm_dir: str | Path = "models/final",
    skip_training: bool = False,
) -> Dict[str, Any]:
    """Train the final GRU, fit its calibrator, weight and policy; write artifacts and tables."""
    if cfg is None:
        cfg = load_config()
    out_dir, lgbm_dir = Path(out_dir), Path(lgbm_dir)
    tables = Path(cfg["paths"]["tables"])
    with open(tables / "ensemble_dev_selection.json") as f:
        selection = json.load(f)
    variant = selection["chosen_variant"]
    oof_path = Path(selection["variants"][variant])
    with open(oof_path.parent / "meta.json") as f:
        gru_meta = json.load(f)

    epochs = final_epochs(gru_meta)
    alpha = float(gru_meta["selected_alpha"])
    cfg["gru"].update(gru_meta.get("gru_config", {}))
    cfg["gru"]["focal_alpha"] = alpha  # the alpha actually used, not the config default
    top_k = int(gru_meta.get("extra_top_k", 0))
    extra = top_lgbm_features(lgbm_dir / "lgbm.txt", top_k) if top_k else []
    logger.info(
        "Final GRU: variant %s, %d epochs, alpha %.2f, %d extra", variant, epochs, alpha, len(extra)
    )
    if not skip_training:
        train_final(cfg, epochs, alpha, out_dir / "gru.pt", extra_features=extra)

    df = load_oof(str(oof_path))
    labels = df["SepsisLabel"].to_numpy(int)
    lgbm_cal = joblib.load(lgbm_dir / "calibrator.joblib")
    cal_l = np.clip(lgbm_cal.predict(df["lgbm"].to_numpy()), 0.0, 1.0)
    refit_gap = float(
        np.abs(cal_l - _fit_iso(df["lgbm"].to_numpy(), labels).predict(df["lgbm"].to_numpy())).max()
    )
    logger.info("Frozen LightGBM calibrator vs isotonic refit on the same OOF: %.2e", refit_gap)
    gru_iso = _fit_iso(df["gru"].to_numpy(), labels)
    cal_g = gru_iso.predict(df["gru"].to_numpy())

    weight = float(selection["pooled_weight"])
    if weight not in WEIGHTS:
        raise ValueError(f"Weight {weight} is outside the fixed grid {WEIGHTS}")
    blend = df[ID_COLS].assign(p=weight * cal_l + (1 - weight) * cal_g)
    policy, _ = search_policy(blend, "p", cfg)
    blend["a"] = apply_policy(blend, "p", policy["threshold"], policy["k"], policy["sticky"])
    pooled = evaluate_predictions(blend, "p", "a", cfg)

    comp = pd.read_csv(tables / "ensemble_dev_comparison.csv")
    nested_row = comp[
        (comp["model"] == "ensemble")
        & (comp["gru_variant"] == variant)
        & (comp["scheme"] == "nested")
    ].iloc[0]
    nested = {k: float(nested_row[k]) for k in METRICS}

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gru_iso, out_dir / "gru_calibrator.joblib")
    ensemble = {
        "kind": "lgbm+gru",
        "weight_lgbm": weight,
        "weight_gru": round(1 - weight, 10),
        "policy": {k: policy[k] for k in ("threshold", "k", "sticky")},
        "gru_variant": variant,
        "gru_epochs": epochs,
        "gru_best_epochs_cv": [f["best_epoch"] for f in gru_meta["folds"]],
        "focal_alpha": alpha,
        "gru_config": dict(cfg["gru"]),
        "extra_features": extra,
        "lgbm_dir": str(lgbm_dir).replace("\\", "/"),
        "git_sha": git_sha(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": "development split only (32,268 patients); lockbox not used",
    }
    with open(out_dir / "ensemble.json", "w") as f:
        json.dump(ensemble, f, indent=2)
    dev_metrics = {
        "label": "development, nested 5-fold CV",
        "nested": nested,
        "pooled_in_sample": pooled,
        "pooled_note": "calibration, weight and policy fit on the same OOF rows (optimistic)",
        "policy_dev_utility": policy["utility_norm"],
        "lgbm_calibrator_refit_gap": refit_gap,
    }
    with open(out_dir / "dev_metrics.json", "w") as f:
        json.dump(dev_metrics, f, indent=2)

    rows = []
    for scheme, m in (("nested", nested), ("pooled", pooled)):
        rows.append(
            {
                "model": "ensemble",
                "gru_variant": variant,
                "scheme": scheme,
                **{k: m[k] for k in METRICS},
                "weight_lgbm": weight,
                **ensemble["policy"],
                "gru_epochs": epochs,
            }
        )
    pd.DataFrame(rows).to_csv(tables / "ensemble_dev_final.csv", index=False)
    print(json.dumps({**ensemble, "dev_metrics": dev_metrics}, indent=2))
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the final dev-only LightGBM + GRU ensemble")
    parser.add_argument("--out-dir", default="models/ensemble_final")
    parser.add_argument("--skip-training", action="store_true", help="Reuse an existing gru.pt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_final_ensemble(out_dir=args.out_dir, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
