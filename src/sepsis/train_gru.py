"""GRU 5-fold dev training for SepsisSentinel — P6.T2, plus the final dev-only GRU.

Same folds as LightGBM, so OOF predictions compare directly. The lockbox is not
used: sequences.py only contains dev patients. The GRU is unidirectional, so an
hour's output depends only on that hour and earlier ones.

Variants (fixed before any variant result was seen): `--hidden 64`, and
`--extra-top-k 32`, which appends the 32 highest-gain LightGBM features. For fold k
those come from models/lgbm_tuned/fold{k}.txt, trained without fold k.
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from sepsis.config import load_config
from sepsis.constants import SEED
from sepsis.sequences import input_columns, standardize, standardize_extra

logger = logging.getLogger(__name__)


class SepsisGRU(nn.Module):
    """Input projection, unidirectional GRU, per-hour logit head."""

    def __init__(self, n_inputs: int, cfg: Dict[str, Any]) -> None:
        super().__init__()
        proj = cfg.get("input_proj", 128)
        hidden = cfg.get("hidden", 128)
        dropout = cfg.get("dropout", 0.2)
        self.proj = nn.Sequential(nn.Linear(n_inputs, proj), nn.ReLU())
        self.gru = nn.GRU(
            proj,
            hidden,
            num_layers=cfg.get("layers", 2),
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.proj(x))
        return self.head(out).squeeze(-1)


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float
) -> torch.Tensor:
    """Per-element focal loss (Lin et al. 2017), computed from logits."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t) ** gamma * bce


def pad_batch(
    X: np.ndarray, y: np.ndarray, offsets: np.ndarray, idx: np.ndarray, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad the patients in idx; mask is True on real hours."""
    lengths = np.minimum(offsets[idx + 1] - offsets[idx], max_len)
    T = int(lengths.max())
    xb = np.zeros((len(idx), T, X.shape[1]), dtype="float32")
    yb = np.zeros((len(idx), T), dtype="float32")
    mask = np.zeros((len(idx), T), dtype=bool)
    for i, (p, n) in enumerate(zip(idx, lengths)):
        start = offsets[p]
        xb[i, :n] = X[start : start + n]
        yb[i, :n] = y[start : start + n]
        mask[i, :n] = True
    return torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(mask)


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    offsets: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Probabilities for every hour of the patients in idx, in row order."""
    model.eval()
    probs = np.full(offsets[-1], np.nan, dtype="float64")
    order = idx[np.argsort(offsets[idx + 1] - offsets[idx])]
    for s in range(0, len(order), batch_size):
        batch = order[s : s + batch_size]
        xb, _, mask = pad_batch(X, y, offsets, batch, 10**9)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(xb.to(device, non_blocking=True))
        p = torch.sigmoid(logits.float()).cpu().numpy()
        m = mask.numpy()
        for i, patient in enumerate(batch):
            probs[offsets[patient] : offsets[patient + 1]] = p[i][m[i]]
    return probs


def train_epoch(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    X: np.ndarray,
    y: np.ndarray,
    offsets: np.ndarray,
    order: np.ndarray,
    alpha: float,
    gcfg: Dict[str, Any],
    device: torch.device,
) -> float:
    """One pass over the patients in order; returns mean focal loss per real hour."""
    model.train()
    total, hours = 0.0, 0
    for s in range(0, len(order), gcfg["batch_size"]):
        xb, yb, mask = pad_batch(X, y, offsets, order[s : s + gcfg["batch_size"]], gcfg["max_len"])
        xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss_all = focal_loss(model(xb).float(), yb, alpha, gcfg["focal_gamma"])
            loss = loss_all[mask].mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        nn.utils.clip_grad_norm_(model.parameters(), gcfg["grad_clip"])
        scaler.step(optim)
        scaler.update()
        total += loss.item() * int(mask.sum())
        hours += int(mask.sum())
    return total / hours


def top_lgbm_features(model_path: str | Path, k: int) -> List[str]:
    """Highest-gain LightGBM features that are not already GRU inputs."""
    booster = lgb.Booster(model_file=str(model_path))
    names = np.array(booster.feature_name())
    gain = booster.feature_importance("gain")
    skip = set(input_columns()) | {"hour"}
    return [n for n in names[np.argsort(-gain, kind="stable")] if n not in skip][:k]


def load_feature_block(
    cfg: Dict[str, Any], columns: Sequence[str], patients: pd.DataFrame, hours: np.ndarray
) -> np.ndarray:
    """Feature columns for the sequence rows, in sequence row order (dev patients only)."""
    df = pd.read_parquet(
        cfg["paths"]["features"],
        columns=["patient_id", "hour", *columns],
        filters=[("patient_id", "in", patients["patient_id"].tolist())],
    ).sort_values(["patient_id", "hour"])
    if len(df) != len(hours) or not np.array_equal(df["hour"].to_numpy(), hours):
        raise ValueError("Feature rows do not line up with the sequence rows")
    return df[list(columns)].to_numpy("float32")


def train_one_fold(
    X_raw: np.ndarray,
    y: np.ndarray,
    offsets: np.ndarray,
    folds: np.ndarray,
    fold: int,
    alpha: float,
    cfg: Dict[str, Any],
    device: torch.device,
    seed: int = SEED,
    extra: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train on folds != fold, early-stop on fold's AUPRC, return its predictions."""
    gcfg = cfg["gru"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + fold)

    n_rows_per_patient = offsets[1:] - offsets[:-1]
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]
    train_rows = np.repeat(folds != fold, n_rows_per_patient)
    val_rows = np.repeat(folds == fold, n_rows_per_patient)
    X, norm = standardize(X_raw, train_rows)
    if extra is not None:
        E, norm["extra"] = standardize_extra(extra, train_rows)
        X = np.hstack([X, E])

    model = SepsisGRU(X.shape[1], gcfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best: Dict[str, Any] = {"auprc": -1.0, "epoch": -1, "state": None}
    history: List[Dict[str, float]] = []
    for epoch in range(gcfg["max_epochs"]):
        order = rng.permutation(train_idx)
        loss = train_epoch(model, optim, scaler, X, y, offsets, order, alpha, gcfg, device)
        val_probs = predict(model, X, y, offsets, val_idx, device)[val_rows]
        auprc = float(average_precision_score(y[val_rows], val_probs))
        sched.step(auprc)
        history.append({"epoch": epoch, "loss": loss, "val_auprc": auprc})
        logger.info(
            "fold %d alpha %.2f epoch %d loss %.5f val AUPRC %.4f",
            fold, alpha, epoch, loss, auprc,
        )
        if auprc > best["auprc"]:
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best = {"auprc": auprc, "epoch": epoch, "state": state}
        elif epoch - best["epoch"] >= gcfg["patience"]:
            break

    model.load_state_dict(best["state"])
    probs = predict(model, X, y, offsets, val_idx, device)
    return {
        "fold": fold,
        "alpha": alpha,
        "best_epoch": best["epoch"],
        "auprc": best["auprc"],
        "auroc": float(roc_auc_score(y[val_rows], probs[val_rows])),
        "val_rows": val_rows,
        "probs": probs,
        "state": best["state"],
        "norm": norm,
        "history": history,
    }


def load_sequences(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Dev-only sequence arrays written by sequences.py."""
    seq_dir = Path(cfg["paths"]["features"]).parent / "sequences"
    return {
        "X": np.load(seq_dir / "X.npy"),
        "y": np.load(seq_dir / "y.npy"),
        "hours": np.load(seq_dir / "hours.npy"),
        "offsets": np.load(seq_dir / "offsets.npy"),
        "patients": pd.read_csv(seq_dir / "patients.csv"),
    }


def _device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    return device


def run_cv(
    cfg: Optional[Dict[str, Any]] = None,
    output_dir: str = "models/gru",
    alpha: Optional[float] = None,
    extra_top_k: int = 0,
) -> pd.DataFrame:
    """Train 5 folds and save OOF. Without a fixed alpha, pick it on fold 0 first."""
    if cfg is None:
        cfg = load_config()
    seq = load_sequences(cfg)
    X_raw, y, hours, offsets = seq["X"], seq["y"], seq["hours"], seq["offsets"]
    patients = seq["patients"]
    folds = patients["fold"].to_numpy()
    device = _device()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_ids = sorted(int(f) for f in set(folds))
    extra_cols: Dict[int, List[str]] = {}
    extra_all: Optional[np.ndarray] = None
    union: List[str] = []
    if extra_top_k:
        lgbm_dir = Path(cfg["paths"]["models"]) / "lgbm_tuned"
        extra_cols = {
            f: top_lgbm_features(lgbm_dir / f"fold{f}.txt", extra_top_k) for f in fold_ids
        }
        union = sorted(set().union(*extra_cols.values()))
        extra_all = load_feature_block(cfg, union, patients, hours)

    def extra_for(fold: int) -> Optional[np.ndarray]:
        if extra_all is None:
            return None
        return extra_all[:, [union.index(c) for c in extra_cols[fold]]]

    alpha_runs: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    if alpha is None:
        alpha_runs = [
            train_one_fold(X_raw, y, offsets, folds, 0, a, cfg, device, extra=extra_for(0))
            for a in cfg["gru"].get("alpha_grid", [0.5, 0.75, 0.9])
        ]
        best_run = max(alpha_runs, key=lambda r: r["auprc"])
        alpha = float(best_run["alpha"])
        logger.info("Selected focal alpha %.2f on fold 0", alpha)
        results.append(best_run)
    for fold in fold_ids:
        if any(r["fold"] == fold for r in results):
            continue
        results.append(
            train_one_fold(
                X_raw, y, offsets, folds, fold, alpha, cfg, device, extra=extra_for(fold)
            )
        )

    oof = np.full(len(y), np.nan)
    for r in results:
        oof[r["val_rows"]] = r["probs"][r["val_rows"]]
        torch.save(
            {
                "state": r["state"],
                "norm": r["norm"],
                "alpha": alpha,
                "n_inputs": X_raw.shape[1] + len(extra_cols.get(r["fold"], [])),
                "config": dict(cfg["gru"]),
                "extra_features": extra_cols.get(r["fold"], []),
            },
            out_dir / f"fold{r['fold']}.pt",
        )
    if np.isnan(oof).any():
        raise RuntimeError("Missing OOF predictions")

    rows = np.repeat(patients["patient_id"].to_numpy(), offsets[1:] - offsets[:-1])
    oof_df = pd.DataFrame({"patient_id": rows, "hour": hours, "SepsisLabel": y, "prob": oof})
    oof_df.to_parquet(out_dir / "oof.parquet", index=False)

    keys = ("fold", "alpha", "best_epoch", "auprc", "auroc", "history")
    meta = {
        "selected_alpha": alpha,
        "gru_config": dict(cfg["gru"]),
        "extra_top_k": extra_top_k,
        "extra_features": {str(f): c for f, c in extra_cols.items()},
        "alpha_check": [{k: r[k] for k in keys} for r in alpha_runs],
        "folds": [{k: r[k] for k in keys} for r in sorted(results, key=lambda r: r["fold"])],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return oof_df


def git_sha() -> str:
    """Short commit hash of HEAD, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_final(
    cfg: Dict[str, Any],
    epochs: int,
    alpha: float,
    out_path: str | Path,
    extra_features: Sequence[str] = (),
    seed: int = SEED,
) -> Dict[str, Any]:
    """Train on all dev patients for a fixed number of epochs (no validation data)."""
    gcfg = cfg["gru"]
    seq = load_sequences(cfg)
    X_raw, y, offsets, patients = seq["X"], seq["y"], seq["offsets"], seq["patients"]
    device = _device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    all_rows = np.ones(len(y), dtype=bool)
    X, norm = standardize(X_raw, all_rows)
    if extra_features:
        E_raw = load_feature_block(cfg, list(extra_features), patients, seq["hours"])
        E, norm["extra"] = standardize_extra(E_raw, all_rows)
        X = np.hstack([X, E])

    model = SepsisGRU(X.shape[1], gcfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    idx = np.arange(len(patients))
    history = []
    for epoch in range(epochs):
        order = rng.permutation(idx)
        loss = train_epoch(model, optim, scaler, X, y, offsets, order, alpha, gcfg, device)
        history.append({"epoch": epoch, "loss": loss})
        logger.info("final GRU epoch %d loss %.5f", epoch, loss)

    bundle = {
        "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "norm": norm,
        "alpha": alpha,
        "n_inputs": int(X.shape[1]),
        "config": dict(gcfg),
        "extra_features": list(extra_features),
        "epochs": epochs,
        "n_patients": int(len(patients)),
        "n_rows": int(len(y)),
        "history": history,
        "git_sha": git_sha(),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    logger.info("Saved final GRU (%d patients, %d epochs) to %s", len(patients), epochs, out_path)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GRU on dev folds")
    parser.add_argument("--output-dir", default="models/gru")
    parser.add_argument("--alpha", type=float, default=None, help="Fixed focal alpha")
    parser.add_argument("--hidden", type=int, default=None, help="Override the GRU hidden size")
    parser.add_argument(
        "--extra-top-k", type=int, default=0, help="Append the k highest-gain LightGBM features"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    if args.hidden is not None:
        cfg["gru"]["hidden"] = args.hidden
    run_cv(cfg, output_dir=args.output_dir, alpha=args.alpha, extra_top_k=args.extra_top_k)


if __name__ == "__main__":
    main()
