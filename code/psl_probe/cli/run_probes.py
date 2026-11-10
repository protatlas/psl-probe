"""Stage 3: fit probes across every layer, pooling strategy and control task.

Produces ``results/probes/<tag>_probe_results.csv`` with one row per
(layer, pooling, task) combination, plus a per-class table for the compartment
breakdown figures.

Example
-------
::

    python -m psl_probe.cli.run_probes --config configs/esm2_650m.yaml
    python -m psl_probe.cli.run_probes --layers 0 6 12 18 24 30 33 --set probe.kind=mlp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from psl_probe.cli import add_common_arguments, resolve
from psl_probe.config import REPO_ROOT, Config
from psl_probe.data.curate import load_dataset
from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.embeddings.extract import embedding_metadata, load_pooled
from psl_probe.probes.control import make_control_labels
from psl_probe.probes.trainer import train_probe
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def _align(dataset: pd.DataFrame, accessions: list[str]) -> pd.DataFrame:
    """Reorder the benchmark table to match the embedding row order."""
    indexed = dataset.set_index("accession")
    missing = set(accessions) - set(indexed.index)
    if missing:
        raise KeyError(f"{len(missing)} embedded accessions are absent from the benchmark table")
    return indexed.loc[accessions].reset_index()


def _split_indices(
    dataset: pd.DataFrame, cfg: Config, val_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays for train / validation / test under the homology partition.

    The last fold is the held-out test partition; ``val_fold`` selects one of the
    remaining folds for validation and early stopping.
    """
    test_fold = cfg.data.n_folds - 1
    if val_fold == test_fold:
        raise ValueError("val_fold must differ from the reserved test fold")
    folds = dataset["fold"].to_numpy()
    return (
        np.flatnonzero((folds != test_fold) & (folds != val_fold)),
        np.flatnonzero(folds == val_fold),
        np.flatnonzero(folds == test_fold),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--embeddings", type=Path, default=None, help="Override embedding HDF5")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Layer subset")
    parser.add_argument("--val-fold", type=int, default=0, help="Fold used for validation")
    parser.add_argument("--tag-suffix", default="", help="Suffix for output filenames")
    parser.add_argument("--no-control", action="store_true", help="Skip control tasks")
    parser.add_argument(
        "--seeds", type=int, default=1, metavar="N",
        help=(
            "Fit the real task under N seeds per layer so the reported curve carries "
            "probe-fitting variance. Control tasks already vary their fit seed across "
            "their control_seeds label draws, so they are unaffected."
        ),
    )
    args = parser.parse_args()
    cfg = resolve(args, "run_probes")

    # A relative --embeddings is resolved against the repository root, matching
    # how every configured path behaves; otherwise it would silently resolve
    # against the working directory and fail from anywhere but the repo root.
    embedding_path = args.embeddings or (cfg.paths.embeddings / f"{cfg.model.tag}.h5")
    if not embedding_path.is_absolute():
        embedding_path = REPO_ROOT / embedding_path
    metadata = embedding_metadata(embedding_path)
    logger.info("Probing %s (%s)", embedding_path.name, metadata.get("model_id"))

    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")
    class_names = list(LOCALIZATION_CLASSES)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    records: list[dict] = []
    per_class_records: list[dict] = []

    for strategy in cfg.pooling.strategies:
        features_all, accessions = load_pooled(embedding_path, strategy)
        aligned = _align(dataset, accessions)
        train_index, val_index, test_index = _split_indices(aligned, cfg, args.val_fold)
        y_real = aligned[class_names].to_numpy(dtype=np.float32)

        # Each entry is (task name, labels, fit seed). The real task is repeated
        # under several fit seeds with identical labels, which isolates probe
        # initialisation and batch order. Each control draw already varies both its
        # labels and its fit seed, so it samples the same variance component.
        n_real_seeds = max(args.seeds, 1)
        tasks: list[tuple[str, np.ndarray, int]] = [
            ("real", y_real, cfg.seed + i) for i in range(n_real_seeds)
        ]
        if cfg.probe.control_task and not args.no_control:
            tasks += [
                (
                    "control",
                    make_control_labels(aligned, class_names, seed=cfg.seed + s),
                    cfg.seed + s,
                )
                for s in range(cfg.probe.control_seeds)
            ]

        n_layers = features_all.shape[1]
        layers = args.layers if args.layers else list(range(n_layers))

        for layer in tqdm(layers, desc=f"probing {strategy}", unit="layer"):
            features = features_all[:, layer, :]
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
                records.append(
                    result.to_record(
                        model=cfg.model.tag, pooling=strategy, layer=layer,
                        task=task_name, task_seed=fit_seed, probe=cfg.probe.kind,
                    )
                )
                if task_name == "real":
                    for row in result.per_class:
                        per_class_records.append(
                            {"model": cfg.model.tag, "pooling": strategy, "layer": layer,
                             "seed": fit_seed, **row}
                        )

    results = pd.DataFrame(records)
    suffix = args.tag_suffix
    out_results = cfg.paths.probes / f"{cfg.model.tag}{suffix}_probe_results.csv"
    out_per_class = cfg.paths.probes / f"{cfg.model.tag}{suffix}_per_class.csv"
    results.to_csv(out_results, index=False)
    pd.DataFrame(per_class_records).to_csv(out_per_class, index=False)

    selectivity = compute_selectivity(results)
    selectivity.to_csv(cfg.paths.probes / f"{cfg.model.tag}{suffix}_selectivity.csv", index=False)

    logger.info("Wrote %s and %s", out_results, out_per_class)


def compute_selectivity(results: pd.DataFrame, metric: str = "macro_mcc") -> pd.DataFrame:
    """Selectivity = real-task score minus mean control-task score, per layer.

    Following Hewitt & Liang (2019); a high score with low selectivity indicates the
    probe is exploiting its own capacity rather than reading the representation.
    """
    keys = ["model", "pooling", "layer"]
    real = (
        results[results["task"] == "real"]
        .groupby(keys)[metric]
        .agg(real="mean", real_sd="std", n_real_seeds="size")
        .reset_index()
    )
    control = results[results["task"] == "control"]
    if control.empty:
        real["control"] = np.nan
        real["control_sd"] = np.nan
        real["selectivity"] = np.nan
        return real

    control_stats = (
        control.groupby(keys)[metric]
        .agg(control="mean", control_sd="std")
        .reset_index()
    )
    merged = real.merge(control_stats, on=keys, how="left")
    merged["selectivity"] = merged["real"] - merged["control"]
    # Independent sources: fit variance of the real probe, label+fit variance of
    # the controls. Their variances add for the difference.
    merged["selectivity_sd"] = np.sqrt(
        merged["real_sd"].fillna(0) ** 2 + merged["control_sd"].fillna(0) ** 2
    )
    return merged


if __name__ == "__main__":
    main()
