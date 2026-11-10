"""Homology-aware dataset partitioning.

Random splits catastrophically over-estimate probe accuracy on protein data,
because near-identical homologues land on both sides of the split and the probe
can succeed by nearest-neighbour recall rather than by reading a sorting signal.
We therefore cluster the benchmark with MMseqs2 at a low identity threshold and
assign whole clusters to folds, so no test protein shares >``identity`` sequence
identity with any training protein.

Cluster-to-fold assignment uses a greedy label-balancing heuristic: clusters are
visited from largest to smallest and each is placed in the fold whose current
per-class counts are furthest below their targets. This keeps the (highly
imbalanced, multi-label) class distribution comparable across folds without
breaking homology containment.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.utils.io import write_fasta
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


class MMseqsNotFoundError(RuntimeError):
    """Raised when the ``mmseqs`` executable is unavailable."""


def _require_mmseqs() -> str:
    executable = shutil.which("mmseqs")
    if executable is None:
        raise MMseqsNotFoundError(
            "mmseqs2 not found on PATH. Install it with `conda install -c bioconda mmseqs2` "
            "or see https://github.com/soedinglab/MMseqs2."
        )
    return executable


def cluster_sequences(
    dataset: pd.DataFrame,
    identity: float = 0.30,
    coverage: float = 0.80,
    workdir: Path | str | None = None,
    threads: int = 8,
) -> pd.Series:
    """Cluster sequences with ``mmseqs easy-cluster``.

    Parameters
    ----------
    dataset:
        Benchmark table with ``accession`` and ``sequence`` columns.
    identity:
        Minimum sequence identity within a cluster (``--min-seq-id``).
    coverage:
        Minimum bidirectional coverage (``-c`` with ``--cov-mode 0``).
    workdir:
        Directory for MMseqs2 intermediates. A temporary directory is used and
        removed when ``None``.
    threads:
        CPU threads handed to MMseqs2.

    Returns
    -------
    pandas.Series
        Cluster representative accession, indexed by member accession.
    """
    executable = _require_mmseqs()

    context = tempfile.TemporaryDirectory() if workdir is None else None
    base = Path(context.name) if context is not None else Path(workdir)
    base.mkdir(parents=True, exist_ok=True)

    try:
        fasta = base / "sequences.fasta"
        write_fasta(zip(dataset["accession"], dataset["sequence"], strict=True), fasta)

        prefix = base / "clu"
        command = [
            executable, "easy-cluster",
            str(fasta), str(prefix), str(base / "tmp"),
            "--min-seq-id", str(identity),
            "-c", str(coverage),
            "--cov-mode", "0",
            "--cluster-mode", "0",   # set-cover, the MMseqs2 default
            "--threads", str(threads),
            "-v", "1",
        ]
        logger.info("Running: %s", " ".join(command))
        subprocess.run(command, check=True, capture_output=True, text=True)

        tsv = prefix.with_name(prefix.name + "_cluster.tsv")
        mapping = pd.read_csv(tsv, sep="\t", header=None, names=["representative", "member"])
        series = mapping.set_index("member")["representative"]
        logger.info(
            "MMseqs2 produced %d clusters from %d sequences at %.0f%% identity",
            series.nunique(), len(series), identity * 100,
        )
        return series.reindex(dataset["accession"]).fillna(pd.Series(
            dataset["accession"].values, index=dataset["accession"].values
        ))
    finally:
        if context is not None:
            context.cleanup()


def assign_folds(
    dataset: pd.DataFrame,
    clusters: pd.Series,
    n_folds: int = 5,
    seed: int = 0,
) -> pd.Series:
    """Assign homology clusters to folds with greedy multi-label balancing.

    Returns
    -------
    pandas.Series
        Integer fold index in ``[0, n_folds)``, aligned to ``dataset.index``.
    """
    rng = np.random.default_rng(seed)
    labels = dataset[list(LOCALIZATION_CLASSES)].to_numpy(dtype=np.float64)
    cluster_ids = clusters.reindex(dataset["accession"]).to_numpy()

    # Aggregate label mass and size per cluster.
    frame = pd.DataFrame(labels, columns=list(LOCALIZATION_CLASSES))
    frame["cluster"] = cluster_ids
    per_cluster = frame.groupby("cluster").sum()
    sizes = frame.groupby("cluster").size()

    # Visit large clusters first; break ties randomly for reproducible fairness.
    order = sizes.sort_values(ascending=False, kind="mergesort").index.to_numpy()
    tie_noise = rng.permutation(len(order))
    order = order[np.argsort(-sizes.loc[order].to_numpy() + tie_noise * 1e-9, kind="mergesort")]

    targets = per_cluster.sum(axis=0).to_numpy() / n_folds
    fold_counts = np.zeros((n_folds, len(LOCALIZATION_CLASSES)), dtype=np.float64)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    assignment: dict[str, int] = {}

    for cluster in order:
        contribution = per_cluster.loc[cluster].to_numpy()
        # Deficit-driven choice: prefer the fold most short of its per-class target,
        # with cluster size as the tie-breaker so folds stay similar in size.
        deficits = ((targets - fold_counts) * contribution).sum(axis=1)
        best = np.flatnonzero(deficits == deficits.max())
        chosen = int(best[np.argmin(fold_sizes[best])])
        assignment[cluster] = chosen
        fold_counts[chosen] += contribution
        fold_sizes[chosen] += int(sizes.loc[cluster])

    folds = pd.Series([assignment[c] for c in cluster_ids], index=dataset.index, name="fold")
    logger.info("Fold sizes: %s", folds.value_counts().sort_index().to_dict())
    return folds


def make_splits(
    dataset: pd.DataFrame,
    identity: float = 0.30,
    coverage: float = 0.80,
    n_folds: int = 5,
    seed: int = 0,
    workdir: Path | str | None = None,
    threads: int = 8,
) -> pd.DataFrame:
    """Add ``cluster`` and ``fold`` columns to the benchmark table.

    The outermost fold (``fold == n_folds - 1``) is reserved as the held-out test
    partition; the remaining folds provide cross-validated training/validation.
    """
    clusters = cluster_sequences(
        dataset, identity=identity, coverage=coverage, workdir=workdir, threads=threads
    )
    out = dataset.copy()
    out["cluster"] = clusters.reindex(out["accession"]).to_numpy()
    out["fold"] = assign_folds(out, clusters, n_folds=n_folds, seed=seed).to_numpy()
    out["split"] = np.where(out["fold"] == n_folds - 1, "test", "train")
    return out


def split_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Per-fold, per-class counts, used to verify the balancing worked."""
    rows = []
    for fold, group in dataset.groupby("fold"):
        row = {"fold": int(fold), "n_proteins": len(group), "n_clusters": group["cluster"].nunique()}
        row.update({cls: int(group[cls].sum()) for cls in LOCALIZATION_CLASSES})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
