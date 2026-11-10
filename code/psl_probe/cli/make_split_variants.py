"""Appendix: build alternative fold assignments to quantify homology leakage.

The probing benchmark is partitioned by clustering at 30% sequence identity and
assigning whole clusters to folds. This script regenerates the *same* proteins
under weaker or absent homology control, so the layer sweep can be repeated and
the inflation attributable to leakage measured directly.

Only the fold assignment changes: per-protein embeddings are independent of the
split, so each variant is written as its own ``benchmark.parquet`` under a
separate processed directory and probed with the existing stores via
``--set paths.processed=...``. Nothing is re-embedded.

Variants
--------
``id30``
    The partition used in the main text (30% identity), copied for reference.
``id50``, ``id90``
    Weaker homology control: clusters merge only at higher identity, so more
    related sequences are free to fall on opposite sides of the split.
``random``
    No homology control at all -- proteins are assigned to folds individually.
    This is the condition most papers use implicitly, and the gap between it and
    ``id30`` is the quantity of interest.

Example
-------
::

    python -m psl_probe.cli.make_split_variants --config configs/esm2_650m.yaml \\
        --variants id50 id90 random
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from psl_probe.cli import add_common_arguments, resolve
from psl_probe.data.curate import load_dataset, save_dataset
from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.data.splits import assign_folds, cluster_sequences, split_summary
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

#: Variant name to MMseqs2 identity threshold; ``None`` means no clustering.
VARIANTS: dict[str, float | None] = {
    "id30": 0.30,
    "id50": 0.50,
    "id90": 0.90,
    "random": None,
}


def build_variant(
    dataset: pd.DataFrame, name: str, cfg, threads: int, seed: int
) -> pd.DataFrame:
    """Return a copy of ``dataset`` with ``cluster``/``fold``/``split`` recomputed."""
    identity = VARIANTS[name]
    out = dataset.copy()

    if identity is None:
        # Each protein is its own "cluster": folds are assigned per protein, so
        # homologues are free to straddle the train/test boundary.
        out["cluster"] = out["accession"].to_numpy()
        logger.info("[%s] no homology control: %d singleton clusters", name, len(out))
    else:
        clusters = cluster_sequences(
            out, identity=identity, coverage=cfg.data.homology_coverage, threads=threads
        )
        out["cluster"] = clusters.reindex(out["accession"]).to_numpy()
        logger.info(
            "[%s] %d clusters at %.0f%% identity (%.2f proteins/cluster)",
            name, out["cluster"].nunique(), identity * 100,
            len(out) / max(out["cluster"].nunique(), 1),
        )

    folds = assign_folds(
        out, pd.Series(out["cluster"].to_numpy(), index=out["accession"].to_numpy()),
        n_folds=cfg.data.n_folds, seed=seed,
    )
    out["fold"] = folds.to_numpy()
    out["split"] = np.where(out["fold"] == cfg.data.n_folds - 1, "test", "train")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--variants", nargs="+", default=["id50", "id90", "random"],
        choices=sorted(VARIANTS), help="Split variants to generate",
    )
    parser.add_argument("--threads", type=int, default=8, help="Threads for MMseqs2")
    args = parser.parse_args()
    cfg = resolve(args, "make_split_variants")

    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")
    logger.info("Source benchmark: %d proteins", len(dataset))

    rows = []
    for name in args.variants:
        target = cfg.paths.processed.parent / f"processed_{name}"
        target.mkdir(parents=True, exist_ok=True)

        variant = build_variant(dataset, name, cfg, args.threads, cfg.seed)
        save_dataset(variant, target / "benchmark.parquet")
        split_summary(variant).to_csv(target / "split_summary.csv", index=False)

        # The probe stage also reads these for its tables; copy them across so a
        # variant directory is self-contained.
        for extra in ("class_summary.csv", "filter_report.csv", "signal_summary.csv"):
            source = cfg.paths.processed / extra
            if source.exists():
                shutil.copy(source, target / extra)

        test = variant[variant["split"] == "test"]
        rows.append(
            {
                "variant": name,
                "identity": VARIANTS[name],
                "n_clusters": int(variant["cluster"].nunique()),
                "proteins_per_cluster": len(variant) / max(variant["cluster"].nunique(), 1),
                "n_test": len(test),
                **{c: int(test[c].sum()) for c in LOCALIZATION_CLASSES},
            }
        )
        logger.info("[%s] wrote %s", name, target / "benchmark.parquet")

    summary = pd.DataFrame(rows)
    out = cfg.paths.processed / "split_variants_summary.csv"
    summary.to_csv(out, index=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
