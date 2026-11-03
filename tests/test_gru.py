"""Tests for the GRU comparison (Phase 6)."""

import numpy as np
import pytest

from sepsis.sequences import MEAS_START, N_MEASURED, standardize

torch = pytest.importorskip("torch")

from sepsis.train_gru import SepsisGRU, focal_loss, pad_batch  # noqa: E402

CFG = {"input_proj": 16, "hidden": 16, "layers": 2, "dropout": 0.0}


def test_gru_is_causal():
    """Changing a later hour must not change any earlier output."""
    torch.manual_seed(0)
    model = SepsisGRU(8, CFG).eval()
    x = torch.randn(1, 20, 8)
    changed = x.clone()
    changed[0, 12:] += 5.0
    with torch.no_grad():
        a, b = model(x), model(changed)
    assert torch.equal(a[0, :12], b[0, :12])
    assert not torch.equal(a[0, 12:], b[0, 12:])


def test_focal_loss_matches_formula():
    logits = torch.tensor([2.0, -1.0])
    y = torch.tensor([1.0, 0.0])
    loss = focal_loss(logits, y, alpha=0.75, gamma=2.0)
    p = torch.sigmoid(logits)
    expected = torch.stack(
        [-0.75 * (1 - p[0]) ** 2 * torch.log(p[0]), -0.25 * p[1] ** 2 * torch.log(1 - p[1])]
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    y = torch.tensor([1.0])
    easy = focal_loss(torch.tensor([4.0]), y, 0.5, 2.0)
    hard = focal_loss(torch.tensor([0.0]), y, 0.5, 2.0)
    assert easy < hard


def test_pad_batch_masks_padding_and_truncates():
    X = np.arange(12, dtype="float32").reshape(6, 2)
    y = np.array([0, 0, 1, 0, 0, 1], dtype="int8")
    offsets = np.array([0, 2, 6])
    xb, yb, mask = pad_batch(X, y, offsets, np.array([0, 1]), max_len=3)
    assert xb.shape == (2, 3, 2)
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert yb[1].tolist() == [1.0, 0.0, 0.0]
    assert xb[0, 2].sum() == 0


def test_standardize_uses_only_observed_training_values():
    n_cols = MEAS_START + N_MEASURED + 1
    X = np.zeros((4, n_cols), dtype="float32")
    X[:, 0] = [1.0, 3.0, 100.0, np.nan]
    X[:, MEAS_START] = [1.0, 1.0, 1.0, 0.0]
    out, stats = standardize(X, np.array([True, True, False, False]))
    assert stats["mean"][0] == pytest.approx(2.0)
    assert stats["std"][0] == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(-1.0)
    assert out[2, 0] == pytest.approx(98.0)
    assert out[3, 0] == 0.0
