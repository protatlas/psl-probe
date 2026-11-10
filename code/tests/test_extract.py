"""Length bucketing must cover every sequence and respect the token budget."""

from __future__ import annotations

import numpy as np
import pytest

from psl_probe.embeddings.extract import length_bucketed_batches


@pytest.mark.parametrize("budget", [512, 2048, 16384])
def test_every_index_appears_exactly_once(budget: int) -> None:
    lengths = [10, 500, 12, 1000, 11, 30, 900, 45, 220, 77]
    batches = list(length_bucketed_batches(lengths, budget))
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(len(lengths)))


def test_padded_token_budget_is_respected() -> None:
    lengths = [10, 500, 12, 1000, 11, 30, 900]
    for batch in length_bucketed_batches(lengths, 2000):
        padded = max(lengths[i] for i in batch) * len(batch)
        # A single over-long sequence is allowed through as its own batch.
        assert padded <= 2000 or len(batch) == 1


def test_sorting_keeps_padding_waste_low_on_realistic_lengths() -> None:
    """The point of length bucketing: little of the compute is spent on padding.

    A wide batch is legal when the token budget does not bind (four short
    sequences plus one long one still fit), so the property worth asserting is
    aggregate padding efficiency over a realistic length distribution rather than
    a per-batch length spread.
    """
    rng = np.random.default_rng(0)
    # Log-normal lengths approximate the Swiss-Prot length distribution.
    lengths = np.clip(rng.lognormal(mean=5.8, sigma=0.6, size=5000), 40, 4096)
    lengths = lengths.astype(int).tolist()

    batches = list(length_bucketed_batches(lengths, 16384))
    real = sum(lengths)
    padded = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    assert padded / real < 1.10  # under 10% of compute wasted on padding


def test_max_batch_size_is_enforced() -> None:
    lengths = [5] * 500
    for batch in length_bucketed_batches(lengths, 1_000_000, max_batch_size=16):
        assert len(batch) <= 16
