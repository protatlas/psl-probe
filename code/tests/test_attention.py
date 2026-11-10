"""Attention alignment and enrichment-null behaviour."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psl_probe.analysis.attention import (
    attention_enrichment,
    bootstrap_enrichment,
    decoy_masks,
    signal_mask_from_features,
)


def test_signal_mask_uses_one_based_inclusive_uniprot_coordinates() -> None:
    """UniProt SIGNAL 1..22 must mark residues 0..21 of a 0-based mask."""
    mask = signal_mask_from_features(
        [{"kind": "signal_peptide", "start": 1, "end": 22}], sequence_length=100
    )
    assert mask is not None
    assert mask[:22].all() and not mask[22:].any()
    assert mask.sum() == 22


def test_signal_mask_returns_none_without_a_relevant_feature() -> None:
    assert signal_mask_from_features([{"kind": "transmembrane", "start": 5, "end": 25}], 100) is None
    assert signal_mask_from_features([], 100) is None


def test_decoy_is_length_matched_and_downstream() -> None:
    rng = np.random.default_rng(0)
    signal = np.zeros(300, dtype=bool)
    signal[:22] = True
    for _ in range(20):
        decoy = decoy_masks([signal], [300], rng)[0]
        assert decoy is not None
        assert decoy.sum() == signal.sum()          # same length
        assert np.flatnonzero(decoy).min() >= 22    # strictly downstream


def test_decoy_unavailable_when_no_downstream_room() -> None:
    """A signal filling most of a short protein leaves no room for its decoy."""
    rng = np.random.default_rng(0)
    signal = np.zeros(40, dtype=bool)
    signal[:30] = True
    assert decoy_masks([signal], [40], rng)[0] is None


def test_enrichment_is_one_for_uniform_attention() -> None:
    """A head attending uniformly must score exactly 1, the null value."""
    n_res, n_heads = 50, 2
    uniform = torch.full((1, n_heads, n_res, n_res), 1.0 / n_res)
    mask = torch.ones(1, n_res, dtype=torch.bool)
    signal = np.zeros(n_res, dtype=bool)
    signal[:10] = True
    result = attention_enrichment((uniform,), mask, [signal])[0]
    assert result is not None
    assert np.allclose(result, 1.0, atol=1e-5)


def test_enrichment_scales_with_concentrated_attention() -> None:
    """All attention on the signal span gives enrichment = length / signal_length."""
    n_res, n_heads, signal_len = 50, 1, 10
    attn = torch.zeros(1, n_heads, n_res, n_res)
    attn[:, :, :, :signal_len] = 1.0 / signal_len
    mask = torch.ones(1, n_res, dtype=torch.bool)
    signal = np.zeros(n_res, dtype=bool)
    signal[:signal_len] = True
    result = attention_enrichment((attn,), mask, [signal])[0]
    assert np.allclose(result, n_res / signal_len, atol=1e-4)


def test_bootstrap_interval_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    per_protein = [rng.normal(2.0, 0.5, (3, 4)) for _ in range(200)]
    out = bootstrap_enrichment(per_protein, n_samples=300, seed=0)
    assert out["mean"].shape == (3, 4)
    assert np.all(out["lo"] <= out["mean"]) and np.all(out["mean"] <= out["hi"])
    assert int(out["n_proteins"]) == 200


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no per-protein"):
        bootstrap_enrichment([])
