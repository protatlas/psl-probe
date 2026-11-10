"""Control tasks for probe selectivity.

A probe that scores well may be reading structure out of the representation, or
may simply be exploiting the probe's own capacity to memorise. Hewitt & Liang
(EMNLP 2019) address this with a *control task*: the same inputs paired with
random labels that are fixed per input type. Selectivity -- task performance minus
control performance -- credits the representation rather than the classifier.

For proteins the natural "type" is the homology cluster, not the individual
sequence: assigning random labels per cluster means a probe can only fit the
control task by memorising cluster identity, which is exactly the shortcut our
homology-aware splits are designed to expose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def make_control_labels(
    dataset: pd.DataFrame,
    label_columns: list[str],
    seed: int = 0,
    group_column: str = "cluster",
    preserve_marginals: bool = True,
) -> np.ndarray:
    """Generate random labels held constant within each homology cluster.

    Parameters
    ----------
    dataset:
        Benchmark table containing ``group_column`` and the label columns.
    label_columns:
        Names of the binary label columns to randomise.
    seed:
        Seed for the permutation.
    group_column:
        Column defining the input type. Every row sharing a value receives the
        same random label vector.
    preserve_marginals:
        If ``True`` (default), control labels are drawn from the empirical
        per-class prevalence, so the control task has the same class imbalance and
        the same chance-level score as the real task. Without this, selectivity is
        confounded by a difference in task difficulty.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_classes)`` float32 control label matrix.
    """
    if group_column not in dataset.columns:
        raise KeyError(
            f"Control tasks require a {group_column!r} column; run make_splits() first."
        )

    rng = np.random.default_rng(seed)
    groups = dataset[group_column].to_numpy()
    unique_groups, inverse = np.unique(groups, return_inverse=True)

    real = dataset[label_columns].to_numpy(dtype=np.float32)
    if preserve_marginals:
        # Each cluster draws its label independently with probability equal to the
        # real *row-level* prevalence. Because control labels are then broadcast to
        # every member, the expected row-level prevalence of the control task equals
        # that of the real task regardless of how cluster sizes are distributed.
        cluster_labels = np.zeros((len(unique_groups), len(label_columns)), dtype=np.float32)
        for j in range(len(label_columns)):
            probability = float(real[:, j].mean())
            cluster_labels[:, j] = (
                rng.random(len(unique_groups)) < probability
            ).astype(np.float32)
    else:
        cluster_labels = rng.integers(
            0, 2, size=(len(unique_groups), len(label_columns))
        ).astype(np.float32)

    control = cluster_labels[inverse]
    logger.info(
        "Control task (seed=%d): %d clusters, mean labels/protein real=%.2f control=%.2f",
        seed, len(unique_groups), real.sum(axis=1).mean(), control.sum(axis=1).mean(),
    )
    return control
