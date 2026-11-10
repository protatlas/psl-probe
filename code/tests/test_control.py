"""Control tasks must be memorisable only via cluster identity, and equally hard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psl_probe.probes.control import make_control_labels

CLASSES = ["a", "b", "c"]


def _frame(n: int = 20000, n_clusters: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"cluster": rng.integers(0, n_clusters, n)})
    for cls, p in zip(CLASSES, (0.5, 0.2, 0.05), strict=True):
        frame[cls] = (rng.random(n) < p).astype(int)
    return frame


def test_labels_are_constant_within_a_cluster() -> None:
    frame = _frame()
    control = make_control_labels(frame, CLASSES, seed=1)
    for cluster in frame["cluster"].unique()[:200]:
        rows = control[frame["cluster"].to_numpy() == cluster]
        assert np.unique(rows, axis=0).shape[0] == 1


def test_marginals_match_the_real_task() -> None:
    """Equal chance level is what makes selectivity interpretable."""
    frame = _frame()
    control = make_control_labels(frame, CLASSES, seed=1)
    real = frame[CLASSES].to_numpy().mean(axis=0)
    assert np.allclose(control.mean(axis=0), real, atol=0.02)


def test_different_seeds_give_different_labels() -> None:
    frame = _frame(n=2000, n_clusters=300)
    first = make_control_labels(frame, CLASSES, seed=1)
    second = make_control_labels(frame, CLASSES, seed=2)
    assert not np.array_equal(first, second)


def test_missing_cluster_column_raises() -> None:
    frame = pd.DataFrame({"a": [0, 1], "b": [1, 0], "c": [0, 0]})
    with pytest.raises(KeyError, match="cluster"):
        make_control_labels(frame, CLASSES)
