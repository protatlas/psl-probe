"""Stage 3b: fit probes to explicit sequence composition, with no language model.

Answers the question a referee asks of the main result: how much of the decodable
localization signal is recoverable from amino-acid composition alone? The probe,
splits, class weighting, threshold selection, control tasks and multi-seed
protocol are identical to :mod:`psl_probe.cli.run_probes`; only the features
differ, so the numbers are directly comparable to a layer of a pLM.

Produces ``results/probes/composition_baseline{suffix}.csv`` (one row per feature
set, task and seed) and a matching per-class table.

Example
-------
::

    python -m psl_probe.cli.run_composition_baseline --config configs/esm2_650m.yaml --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from psl_probe.analysis.composition import FEATURE_SETS, build_features
from psl_probe.cli import add_common_arguments, resolve
from psl_probe.cli.run_probes import _split_indices, compute_selectivity
from psl_probe.data.curate import load_dataset
from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.probes.control import make_control_labels
from psl_probe.probes.trainer import train_probe
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--feature-sets", nargs="*", default=list(FEATURE_SETS),
        help=f"Subset of {list(FEATURE_SETS)} to fit.",
    )
    parser.add_argument("--val-fold", type=int, default=0, help="Fold used for validation")
    parser.add_argument("--tag-suffix", default="", help="Suffix for output filenames")
    parser.add_argument("--no-control", action="store_true", help="Skip control tasks")
    parser.add_argument(
        "--seeds", type=int, default=5, metavar="N",
        help="Fit the real task under N seeds so the curve carries probe-fitting variance.",
    )
    args = parser.parse_args()
    cfg = resolve(args, "run_composition_baseline")

    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")
    class_names = list(LOCALIZATION_CLASSES)
    train_index, val_index, test_index = _split_indices(dataset, cfg, args.val_fold)
    y_real = dataset[class_names].to_numpy(dtype=np.float32)
    logger.info(
        "Composition baseline on %d proteins (train %d / val %d / test %d)",
        len(dataset), train_index.size, val_index.size, test_index.size,
    )

    # Composition is strongly correlated with homology, so the control task
    # matters more here than for a pLM layer: a probe could in principle score
    # well by recognising the compositional signature of a protein family rather
    # than of a compartment. The control holds labels constant within a cluster,
    # so only cluster memorisation can fit it.
    n_real_seeds = max(args.seeds, 1)
    tasks: list[tuple[str, np.ndarray, int]] = [
        ("real", y_real, cfg.seed + i) for i in range(n_real_seeds)
    ]
    if cfg.probe.control_task and not args.no_control:
        tasks += [
            ("control", make_control_labels(dataset, class_names, seed=cfg.seed + s), cfg.seed + s)
            for s in range(cfg.probe.control_seeds)
        ]

    device = cfg.device if torch.cuda.is_available() else "cpu"
    records: list[dict] = []
    per_class_records: list[dict] = []

    for feature_set in tqdm(args.feature_sets, desc="composition", unit="set"):
        features, _ = build_features(dataset, feature_set)
        for task_name, labels, fit_seed in tasks:
            result = train_probe(
                features[train_index], labels[train_index],
                features[val_index], labels[val_index],
                features[test_index], labels[test_index],
                class_names=class_names,
                cfg=cfg.probe,
                device=device,
                seed=fit_seed,
            )
            # `layer` is carried so the frame shares a schema with the pLM probe
            # results and can be concatenated with them; it is meaningless here
            # and fixed at -1 to make that obvious in the CSV.
            records.append(
                result.to_record(
                    model="composition", pooling=feature_set, layer=-1,
                    task=task_name, task_seed=fit_seed, probe=cfg.probe.kind,
                )
            )
            if task_name == "real":
                for row in result.per_class:
                    per_class_records.append(
                        {"model": "composition", "pooling": feature_set, "layer": -1,
                         "seed": fit_seed, **row}
                    )

    results = pd.DataFrame(records)
    suffix = args.tag_suffix
    out_results = cfg.paths.probes / f"composition_baseline{suffix}.csv"
    results.to_csv(out_results, index=False)
    pd.DataFrame(per_class_records).to_csv(
        cfg.paths.probes / f"composition_baseline{suffix}_per_class.csv", index=False
    )
    compute_selectivity(results).to_csv(
        cfg.paths.probes / f"composition_baseline{suffix}_selectivity.csv", index=False
    )

    summary = (
        results[results["task"] == "real"]
        .groupby("pooling")["macro_mcc"].agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    for name, row in summary.iterrows():
        logger.info("  %-16s macro MCC %.4f +/- %.4f", name, row["mean"], row["std"])
    logger.info("Wrote %s", out_results)


if __name__ == "__main__":
    main()
