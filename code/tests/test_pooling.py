"""Pooling operators must respect the residue mask exactly."""

from __future__ import annotations

import pytest
import torch

from psl_probe.embeddings.pooling import pool_representations


@pytest.fixture
def batch():
    """Two proteins of length 10 and 4 inside a padded tensor of width 10."""
    hidden = torch.arange(3 * 2 * 10 * 4, dtype=torch.float32).reshape(3, 2, 10, 4)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :4] = True
    return hidden, mask


def test_mean_ignores_padding(batch) -> None:
    hidden, mask = batch
    pooled = pool_representations(hidden, mask, ["mean"])["mean"]
    expected = hidden[0, 1, :4, :].mean(dim=0)
    assert torch.allclose(pooled[0, 1], expected)


def test_nterm_window_clipped_to_protein_length(batch) -> None:
    """A 70-residue window on a 4-residue protein must average only 4 residues."""
    hidden, mask = batch
    pooled = pool_representations(hidden, mask, ["nterm"], nterm_window=70)["nterm"]
    expected = hidden[0, 1, :4, :].mean(dim=0)
    assert torch.allclose(pooled[0, 1], expected)


def test_nterm_narrower_than_protein(batch) -> None:
    hidden, mask = batch
    pooled = pool_representations(hidden, mask, ["nterm"], nterm_window=3)["nterm"]
    assert torch.allclose(pooled[0, 0], hidden[0, 0, :3, :].mean(dim=0))


def test_cterm_anchored_to_real_end_not_padding(batch) -> None:
    """The C-terminal window must start from the true end, not the padded end."""
    hidden, mask = batch
    pooled = pool_representations(hidden, mask, ["cterm"], cterm_window=2)["cterm"]
    assert torch.allclose(pooled[0, 1], hidden[0, 1, 2:4, :].mean(dim=0))


def test_max_excludes_padding(batch) -> None:
    hidden, mask = batch
    pooled = pool_representations(hidden, mask, ["max"])["max"]
    assert torch.allclose(pooled[0, 1], hidden[0, 1, :4, :].amax(dim=0))


def test_unknown_strategy_raises(batch) -> None:
    hidden, mask = batch
    with pytest.raises(ValueError, match="Unknown pooling"):
        pool_representations(hidden, mask, ["attention"])


def test_fp16_long_sequence_mean_does_not_overflow() -> None:
    """Regression: summing ~1000 residues of fp16 activations overflowed to inf.

    ESM-2 activations carry a large per-dimension DC component, so the running
    sum over a long protein grows roughly linearly and passes the fp16 ceiling
    (65504) even though the resulting mean is small. The bug corrupted exactly
    the long proteins in the deepest layers and only surfaced later as a NaN
    inside a metric.
    """
    n_residues = 1022
    hidden = torch.full((2, 1, n_residues, 64), 80.0, dtype=torch.float16)
    mask = torch.ones(1, n_residues, dtype=torch.bool)

    # The naive fp16 reduction this test guards against really does overflow.
    assert not torch.isfinite(hidden[0, 0].sum(dim=0)).all()

    pooled = pool_representations(hidden, mask, ["mean"])["mean"]
    assert torch.isfinite(pooled).all()
    assert pooled.dtype == torch.float32
    assert torch.allclose(pooled, torch.full_like(pooled, 80.0), atol=0.1)


def test_all_strategies_return_float32_for_half_input() -> None:
    """Every strategy must promote, or a later one silently reintroduces the bug."""
    hidden = torch.randn(2, 2, 40, 16, dtype=torch.float16)
    mask = torch.ones(2, 40, dtype=torch.bool)
    pooled = pool_representations(hidden, mask, ["mean", "cls", "max", "nterm", "cterm"])
    for name, tensor in pooled.items():
        assert tensor.dtype == torch.float32, f"{name} returned {tensor.dtype}"
