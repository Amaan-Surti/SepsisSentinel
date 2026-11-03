"""GRU 5-fold dev training for SepsisSentinel — P6.T2.

Same folds as LightGBM, so OOF predictions compare directly. The lockbox is not
used: sequences.py only contains dev patients. The GRU is unidirectional, so an
hour's output depends only on that hour and earlier ones.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from sepsis.config import load_config
from sepsis.constants import SEED
from sepsis.sequences import standardize

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

    model = SepsisGRU(X.shape[1], gcfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best: Dict[str, Any] = {"auprc": -1.0, "epoch": -1, "state": None}
    history: List[Dict[str, float]] = []
    for epoch in range(gcfg["max_epochs"]):
        model.train()
        order = rng.permutation(train_idx)
        total, hours = 0.0, 0
        for s in range(0, len(order), gcfg["batch_size"]):
            xb, yb, mask = pad_batch(
                X, y, offsets, order[s : s + gcfg["batch_size"]], gcfg["max_len"]
            )
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

        val_probs = predict(model, X, y, offsets, val_idx, device)[val_rows]
        auprc = float(average_precision_score(y[val_rows], val_probs))
        sched.step(auprc)
        history.append({"epoch": epoch, "loss": total / hours, "val_auprc": auprc})
        logger.info(
            "fold %d alpha %.2f epoch %d loss %.5f val AUPRC %.4f",
            fold, alpha, epoch, total / hours, auprc,
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


def run_cv(cfg: Optional[Dict[str, Any]] = None, output_dir: str = "models/gru") -> pd.DataFrame:
    """Pick focal alpha on fold 0, then train the remaining folds and save OOF."""
    if cfg is None:
        cfg = load_config()
    seq_dir = Path(cfg["paths"]["features"]).parent / "sequences"
    X_raw = np.load(seq_dir / "X.npy")
    y = np.load(seq_dir / "y.npy")
    hours = np.load(seq_dir / "hours.npy")
    offsets = np.load(seq_dir / "offsets.npy")
    patients = pd.read_csv(seq_dir / "patients.csv")
    folds = patients["fold"].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_runs = [
        train_one_fold(X_raw, y, offsets, folds, 0, a, cfg, device)
        for a in cfg["gru"].get("alpha_grid", [0.5, 0.75, 0.9])
    ]
    best_run = max(alpha_runs, key=lambda r: r["auprc"])
    alpha = best_run["alpha"]
    logger.info("Selected focal alpha %.2f on fold 0", alpha)

    results = [best_run]
    for fold in sorted(set(folds) - {0}):
        results.append(train_one_fold(X_raw, y, offsets, folds, int(fold), alpha, cfg, device))

    oof = np.full(len(y), np.nan)
    for r in results:
        oof[r["val_rows"]] = r["probs"][r["val_rows"]]
        torch.save(
            {"state": r["state"], "norm": r["norm"], "alpha": alpha, "n_inputs": X_raw.shape[1]},
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
        "alpha_check": [{k: r[k] for k in keys} for r in alpha_runs],
        "folds": [{k: r[k] for k in keys} for r in results],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return oof_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GRU on dev folds")
    parser.add_argument("--output-dir", default="models/gru")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_cv(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
