"""Stage 4b: sequence-ablation sweep.

For each ablation mode and prefix length this script re-embeds the benchmark,
refits the probe at a fixed layer, and records the per-compartment change in MCC
relative to the unablated baseline. Because the encoder pass dominates the cost,
embeddings for each ablation are cached in ``results/embeddings`` and reused on
re-runs.

Example
-------
::

    python -m psl_probe.cli.run_ablation --layer 24 \
        --modes nterm_delete nterm_mask shuffle_nterm internal_delete
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from psl_probe.analysis.ablation import MODES, ablation_deltas, build_ablated_dataset
from psl_probe.analysis.metrics import multiseed_paired_bootstrap_deltas
from psl_probe.cli import add_common_arguments, resolve
from psl_probe.data.curate import load_dataset
from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.embeddings.extract import extract_embeddings, load_pooled
from psl_probe.embeddings.models import load_plm
from psl_probe.probes.trainer import ProbeResult, train_probe
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def _load_layer(cfg, embedding_path, strategy: str, layer: int, val_fold: int):
    """Load one layer's features plus the aligned labels and split indices.

    Only the requested layer is read from the HDF5 store. Reading every layer and
    then slicing would pull ~6 GB per store into memory as float32, which is
    wasted work when a single layer is probed -- and is paid once per fit, so it
    dominates once several seeds are fitted per condition.
    """
    from psl_probe.cli.run_probes import _align, _split_indices

    features, accessions = load_pooled(embedding_path, strategy, layer=layer)
    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")
    aligned = _align(dataset, accessions)
    splits = _split_indices(aligned, cfg, val_fold)
    labels = aligned[list(LOCALIZATION_CLASSES)].to_numpy(dtype=np.float32)
    return features, labels, splits


def _fit_seeds(cfg, features, labels, splits, seeds: list[int]) -> list[ProbeResult]:
    """Fit the same probe under several seeds on already-loaded features.

    Refitting under multiple seeds is what exposes probe-fitting variance
    (initialisation and batch order) as distinct from test-set sampling variance.
    """
    train_index, val_index, test_index = splits
    device = cfg.device if torch.cuda.is_available() else "cpu"
    return [
        train_probe(
            features[train_index], labels[train_index],
            features[val_index], labels[val_index],
            features[test_index], labels[test_index],
            class_names=list(LOCALIZATION_CLASSES), cfg=cfg.probe,
            device=device, seed=seed,
        )
        for seed in seeds
    ]


def _fit(cfg, embedding_path, strategy: str, layer: int, val_fold: int,
         seeds: list[int] | None = None) -> list[ProbeResult]:
    """Load a layer and fit it under each seed (default: the configured seed only)."""
    features, labels, splits = _load_layer(cfg, embedding_path, strategy, layer, val_fold)
    return _fit_seeds(cfg, features, labels, splits, seeds or [cfg.seed])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--layer", type=int, required=True, help="Layer to probe")
    parser.add_argument("--strategy", default="mean", help="Pooling strategy")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument(
        "--modes", nargs="+",
        default=["nterm_delete", "nterm_mask", "shuffle_nterm", "internal_delete"],
        help="Ablation modes to sweep",
    )
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=None,
        help="Prefix lengths (default: analysis.ablation_prefixes, excluding 0)",
    )
    parser.add_argument(
        "--seeds", type=int, default=1, metavar="N",
        help=(
            "Refit every probe under N seeds and pool their bootstrap draws, so the "
            "reported intervals include probe-fitting variance (initialisation and "
            "batch order) as well as test-set sampling variance. Default 1."
        ),
    )
    parser.add_argument(
        "--contrast", default=None, metavar="MODE:PREFIX",
        help=(
            "Additionally contrast every condition against this one instead of "
            "against the unablated baseline. Use it to isolate the component of a "
            "loss that is specific to the ablated region: contrasting "
            "nterm_delete:70 against internal_delete:70 subtracts the generic cost "
            "of removing the same number of residues elsewhere. Written to "
            "<tag>_ablation_contrast.csv with paired bootstrap intervals."
        ),
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None, metavar="MODE:PREFIX",
        help=(
            "Explicit (mode, prefix) pairs, e.g. nterm_delete:70 internal_delete:70. "
            "Overrides --modes/--prefixes. Use this when only some combinations are "
            "wanted: the cross product of --modes and --prefixes would otherwise "
            "force an encoder pass for every unwanted pair."
        ),
    )
    args = parser.parse_args()
    cfg = resolve(args, "run_ablation")

    if args.conditions:
        conditions = []
        for item in args.conditions:
            if ":" not in item:
                raise ValueError(f"--conditions entries must be MODE:PREFIX, got {item!r}")
            mode, _, prefix = item.partition(":")
            if mode not in MODES:
                raise ValueError(f"Unknown ablation mode {mode!r}; expected one of {MODES}")
            conditions.append((mode, int(prefix)))
    else:
        prefixes = args.prefixes or [p for p in cfg.analysis.ablation_prefixes if p > 0]
        conditions = [(mode, prefix) for mode in args.modes for prefix in prefixes]

    logger.info(
        "Ablation conditions (%d): %s",
        len(conditions), ", ".join(f"{m}{p}" for m, p in conditions),
    )
    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")

    baseline_path = cfg.paths.embeddings / f"{cfg.model.tag}.h5"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"{baseline_path} not found; run extract_embeddings before run_ablation."
        )
    seeds = [cfg.seed + i for i in range(max(args.seeds, 1))]
    logger.info("Fitting each condition under %d seed(s): %s", len(seeds), seeds)

    baselines = _fit(cfg, baseline_path, args.strategy, args.layer, args.val_fold, seeds)
    baseline = baselines[0]
    baseline_macro = np.mean([b.metrics["macro_mcc"] for b in baselines])
    logger.info(
        "Baseline macro MCC at layer %d: %.4f (mean of %d seeds, sd %.4f)",
        args.layer, baseline_macro, len(baselines),
        np.std([b.metrics["macro_mcc"] for b in baselines], ddof=1) if len(baselines) > 1 else 0.0,
    )

    plm = load_plm(cfg.model, device=cfg.device if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    summary: list[dict] = []
    fitted: dict[str, list[ProbeResult]] = {}
    for mode, prefix in conditions:
        tag = f"{mode}{prefix}"
        embedding_path = cfg.paths.embeddings / f"{cfg.model.tag}.{tag}.h5"
        if not embedding_path.exists():
            ablated = build_ablated_dataset(dataset, mode, prefix, seed=cfg.seed)
            extract_embeddings(ablated, plm, cfg, embedding_path)
        else:
            logger.info("Reusing cached embeddings %s", embedding_path.name)

        results = _fit(cfg, embedding_path, args.strategy, args.layer, args.val_fold, seeds)
        result = results[0]
        deltas = ablation_deltas(baseline.per_class, result.per_class, metric="mcc")

        # Paired bootstrap over test proteins, pooled across seeds: the baseline
        # and ablated probes are scored on the same resample (so shared test-set
        # variance cancels) and the draws are pooled over fits (so probe-fitting
        # variance is included). Multiplicity across the ten compartments is
        # controlled by BH q-values and Bonferroni simultaneous intervals.
        intervals = pd.DataFrame(
            multiseed_paired_bootstrap_deltas(
                baseline.test_labels,
                [b.test_scores for b in baselines],
                [r.test_scores for r in results],
                [b.thresholds for b in baselines],
                [r.thresholds for r in results],
                class_names=list(LOCALIZATION_CLASSES),
                n_samples_per_seed=cfg.analysis.bootstrap_samples,
                confidence=cfg.analysis.confidence_level,
                seed=cfg.seed,
            )
        )
        # ``delta`` from the multi-seed path is the seed mean; prefer it over the
        # single-fit value that ablation_deltas computed.
        deltas = deltas.drop(columns=["delta"]).merge(
            intervals, on="class", validate="one_to_one"
        )
        deltas["ablation"] = tag
        deltas["mode"] = mode
        deltas["prefix"] = prefix
        rows.append(deltas)
        fitted[tag] = results

        summary.append(
            {
                "ablation": tag, "mode": mode, "prefix": prefix, "layer": args.layer,
                "pooling": args.strategy,
                **{f"baseline_{k}": v for k, v in baseline.metrics.items()},
                **{f"ablated_{k}": v for k, v in result.metrics.items()},
                "ablated_macro_mcc_seedmean": float(
                    np.mean([r.metrics["macro_mcc"] for r in results])
                ),
                "ablated_macro_mcc_seedsd": float(
                    np.std([r.metrics["macro_mcc"] for r in results], ddof=1)
                ) if len(results) > 1 else 0.0,
                "n_seeds": len(results),
                "delta_macro_mcc": float(
                    np.mean([r.metrics["macro_mcc"] for r in results]) - baseline_macro
                ),
            }
        )
        seed_mean = float(np.mean([r.metrics["macro_mcc"] for r in results]))
        seed_sd = (
            float(np.std([r.metrics["macro_mcc"] for r in results], ddof=1))
            if len(results) > 1 else 0.0
        )
        logger.info(
            "%-18s macro MCC %.4f +/- %.4f (delta %+.4f)",
            tag, seed_mean, seed_sd, seed_mean - baseline_macro,
        )

    if args.contrast:
        mode, _, prefix = args.contrast.partition(":")
        reference_tag = f"{mode}{int(prefix)}"
        if reference_tag not in fitted:
            raise ValueError(
                f"--contrast {args.contrast!r} names {reference_tag}, which is not "
                f"among the conditions that were run: {sorted(fitted)}"
            )
        reference_results = fitted[reference_tag]
        contrast_rows = []
        for tag, results in fitted.items():
            if tag == reference_tag:
                continue
            frame = pd.DataFrame(
                multiseed_paired_bootstrap_deltas(
                    reference_results[0].test_labels,
                    [r.test_scores for r in reference_results],
                    [r.test_scores for r in results],
                    [r.thresholds for r in reference_results],
                    [r.thresholds for r in results],
                    class_names=list(LOCALIZATION_CLASSES),
                    n_samples_per_seed=cfg.analysis.bootstrap_samples,
                    confidence=cfg.analysis.confidence_level,
                    seed=cfg.seed,
                )
            )
            frame["ablation"] = tag
            frame["reference"] = reference_tag
            contrast_rows.append(frame)
        contrast = pd.concat(contrast_rows, ignore_index=True)
        contrast_path = cfg.paths.analysis / f"{cfg.model.tag}_ablation_contrast.csv"
        contrast.to_csv(contrast_path, index=False)
        logger.info("Wrote contrasts against %s to %s", reference_tag, contrast_path)

    per_class = pd.concat(rows, ignore_index=True)
    per_class.to_csv(cfg.paths.analysis / f"{cfg.model.tag}_ablation.csv", index=False)
    pd.DataFrame(summary).to_csv(
        cfg.paths.analysis / f"{cfg.model.tag}_ablation_summary.csv", index=False
    )
    logger.info("Wrote ablation results to %s", cfg.paths.analysis)


if __name__ == "__main__":
    main()
