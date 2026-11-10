"""Fold assignment must contain homology clusters and balance the label load."""

from __future__ import annotations

import numpy as np
import pandas as pd

from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.data.splits import assign_folds, split_summary


def _synthetic(n: int = 2000, n_clusters: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "accession": [f"P{i:05d}" for i in range(n)],
            "cluster": [f"C{c:04d}" for c in rng.integers(0, n_clusters, n)],
        }
    )
    prevalence = np.linspace(0.4, 0.01, len(LOCALIZATION_CLASSES))
    for cls, p in zip(LOCALIZATION_CLASSES, prevalence, strict=True):
        frame[cls] = (rng.random(n) < p).astype(int)
    frame["n_labels"] = frame[list(LOCALIZATION_CLASSES)].sum(axis=1)
    return frame


def test_clusters_are_never_split_across_folds() -> None:
    """This is the property that makes the test split homology-clean."""
    frame = _synthetic()
    clusters = frame.set_index("accession")["cluster"]
    frame["fold"] = assign_folds(frame, clusters, n_folds=5, seed=0).to_numpy()
    per_cluster = frame.groupby("cluster")["fold"].nunique()
    assert (per_cluster == 1).all()


def test_folds_are_roughly_balanced_in_size() -> None:
    frame = _synthetic()
    clusters = frame.set_index("accession")["cluster"]
    frame["fold"] = assign_folds(frame, clusters, n_folds=5, seed=0).to_numpy()
    sizes = frame["fold"].value_counts().to_numpy()
    assert sizes.max() / sizes.min() < 1.5


def test_rare_classes_appear_in_every_fold() -> None:
    """Greedy deficit-driven assignment must not starve a fold of a rare class."""
    frame = _synthetic()
    clusters = frame.set_index("accession")["cluster"]
    frame["fold"] = assign_folds(frame, clusters, n_folds=5, seed=0).to_numpy()
    summary = split_summary(frame)
    rarest = LOCALIZATION_CLASSES[-1]
    assert (summary[rarest] > 0).all()


def test_assignment_is_deterministic_for_a_seed() -> None:
    frame = _synthetic()
    clusters = frame.set_index("accession")["cluster"]
    first = assign_folds(frame, clusters, n_folds=5, seed=7).to_numpy()
    second = assign_folds(frame, clusters, n_folds=5, seed=7).to_numpy()
    assert np.array_equal(first, second)
