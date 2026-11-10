"""Stage 1: build the homology-partitioned benchmark from Swiss-Prot.

Example
-------
::

    python -m psl_probe.cli.prepare_data --config configs/esm2_650m.yaml
    python -m psl_probe.cli.prepare_data --set data.homology_identity=0.5 --limit 20000
"""

from __future__ import annotations

import argparse

from psl_probe.cli import add_common_arguments, resolve
from psl_probe.data.curate import (
    curate_dataset,
    save_dataset,
    signal_feature_summary,
    summarise_dataset,
)
from psl_probe.data.splits import make_splits, split_summary
from psl_probe.data.uniprot import download_swissprot, parse_swissprot
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Parse at most this many Swiss-Prot entries (smoke testing)",
    )
    parser.add_argument("--threads", type=int, default=8, help="Threads for MMseqs2")
    parser.add_argument("--skip-download", action="store_true", help="Use an existing archive")
    args = parser.parse_args()
    cfg = resolve(args, "prepare_data")

    if args.skip_download:
        dat_path = cfg.paths.raw / "uniprot_sprot.dat.gz"
        if not dat_path.exists():
            raise FileNotFoundError(f"--skip-download given but {dat_path} is missing")
    else:
        dat_path = download_swissprot(cfg.paths.raw, release=cfg.data.uniprot_release)

    records = parse_swissprot(
        dat_path,
        require_experimental=cfg.data.require_experimental_evidence,
        kingdoms=cfg.data.kingdoms or None,
        limit=args.limit,
    )
    dataset, filter_report, unmapped = curate_dataset(records, cfg)
    if dataset.empty:
        raise RuntimeError("No proteins survived curation; check the filter settings.")

    dataset = make_splits(
        dataset,
        identity=cfg.data.homology_identity,
        coverage=cfg.data.homology_coverage,
        n_folds=cfg.data.n_folds,
        seed=cfg.seed,
        threads=args.threads,
    )

    save_dataset(dataset, cfg.paths.processed / "benchmark.parquet")
    filter_report.to_csv(cfg.paths.processed / "filter_report.csv", index=False)
    unmapped.to_csv(cfg.paths.processed / "unmapped_terms.csv", index=False)
    summarise_dataset(dataset).to_csv(cfg.paths.processed / "class_summary.csv", index=False)
    signal_feature_summary(dataset).to_csv(cfg.paths.processed / "signal_summary.csv", index=False)
    split_summary(dataset).to_csv(cfg.paths.processed / "split_summary.csv", index=False)

    logger.info("Benchmark ready: %d proteins, %d clusters",
                len(dataset), dataset["cluster"].nunique())


if __name__ == "__main__":
    main()
