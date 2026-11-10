"""Stage 4: layer-similarity, attention alignment and bootstrap confidence intervals.

Example
-------
::

    python -m psl_probe.cli.run_analysis --config configs/esm2_650m.yaml
    python -m psl_probe.cli.run_analysis --skip-attention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from psl_probe.analysis.attention import (
    attention_enrichment,
    bootstrap_enrichment,
    decoy_masks,
    signal_mask_from_features,
    summarise_enrichment,
)
from psl_probe.analysis.cka import layer_cka_matrix
from psl_probe.analysis.metrics import bootstrap_ci, sigmoid
from psl_probe.cli import add_common_arguments, resolve
from psl_probe.config import REPO_ROOT
from psl_probe.data.curate import load_dataset
from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.embeddings.extract import length_bucketed_batches, load_pooled
from psl_probe.embeddings.models import load_plm
from psl_probe.probes.linear import predict_logits
from psl_probe.probes.trainer import train_probe
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def run_cka(cfg, embedding_path: Path, strategy: str) -> Path:
    """Compute and store the layer-by-layer CKA matrix."""
    features, _ = load_pooled(embedding_path, strategy)
    matrix = layer_cka_matrix(
        features, max_samples=cfg.analysis.cka_max_proteins, seed=cfg.seed
    )
    out = cfg.paths.analysis / f"{cfg.model.tag}_{strategy}_cka.npy"
    np.save(out, matrix)
    logger.info("Wrote CKA matrix %s (shape %s)", out, matrix.shape)
    return out


def run_attention(cfg, dataset: pd.DataFrame, n_decoy_draws: int = 5) -> Path:
    """Per-head attention enrichment on curated sorting signals, against a decoy null.

    Attention tensors are never stored to disk -- they are O(L^2) per layer per head
    -- so this pass re-runs the encoder with ``output_attentions=True`` on the subset
    of proteins that carry a curated ``SIGNAL`` or ``TRANSIT`` feature.
    """
    subset = dataset[
        dataset["features"].apply(
            lambda feats: any(
                f["kind"] in {"signal_peptide", "transit_peptide"} for f in feats
            )
        )
    ]
    if subset.empty:
        raise RuntimeError("No proteins with curated SIGNAL/TRANSIT features were found.")
    if len(subset) > cfg.analysis.attention_max_proteins:
        subset = subset.sample(cfg.analysis.attention_max_proteins, random_state=cfg.seed)
    logger.info("Attention analysis over %d proteins with curated signals", len(subset))

    cfg.model.output_attentions = True
    plm = load_plm(cfg.model, device=cfg.device if torch.cuda.is_available() else "cpu")

    sequences = subset["sequence"].tolist()
    features_column = subset["features"].tolist()
    lengths = [min(len(s), cfg.model.max_length) for s in sequences]
    rng = np.random.default_rng(cfg.seed)

    signal_scores: list[np.ndarray] = []
    # One list of per-protein arrays per decoy placement draw. Averaging over
    # several placements stops a single unlucky window from standing in for the
    # whole null.
    decoy_draws: list[list[np.ndarray]] = [[] for _ in range(n_decoy_draws)]
    n_layers = n_heads = 0
    n_unpaired = 0

    # A smaller token budget: attention tensors dominate memory here.
    for batch in tqdm(
        list(length_bucketed_batches(lengths, cfg.model.batch_tokens // 8, max_batch_size=8)),
        desc="attention", unit="batch",
    ):
        batch_sequences = [sequences[i] for i in batch]
        batch_lengths = [lengths[i] for i in batch]
        output = plm.encode(batch_sequences)
        if output.attentions is None:  # pragma: no cover - defensive
            raise RuntimeError("Model did not return attentions; check output_attentions.")
        n_layers = len(output.attentions)
        n_heads = output.attentions[0].shape[1]

        masks = [
            signal_mask_from_features(features_column[i], batch_lengths[j])
            for j, i in enumerate(batch)
        ]
        signal_batch = attention_enrichment(output.attentions, output.mask, masks)
        decoy_batch = [
            attention_enrichment(
                output.attentions, output.mask, decoy_masks(masks, batch_lengths, rng)
            )
            for _ in range(n_decoy_draws)
        ]

        # Keep a protein only if its signal *and* every decoy placement succeeded,
        # and append them at the same list position. Filtering each list
        # independently would leave them misaligned, and the paired contrast below
        # would then subtract one protein's decoy from another protein's signal.
        # A decoy is unavailable when there is no room downstream of the signal for
        # a window of equal length.
        for j in range(len(batch)):
            if signal_batch[j] is None or any(d[j] is None for d in decoy_batch):
                n_unpaired += 1
                continue
            signal_scores.append(signal_batch[j])
            for d in range(n_decoy_draws):
                decoy_draws[d].append(decoy_batch[d][j])

    signal_summary = summarise_enrichment(signal_scores, n_layers, n_heads)
    decoy_scores = [np.nanmean(np.stack(p, axis=0), axis=0)
                    for p in zip(*decoy_draws, strict=True)]
    decoy_summary = summarise_enrichment(decoy_scores, n_layers, n_heads)

    signal_ci = bootstrap_enrichment(
        signal_scores, n_samples=cfg.analysis.bootstrap_samples,
        confidence=cfg.analysis.confidence_level, seed=cfg.seed,
    )
    decoy_ci = bootstrap_enrichment(
        decoy_scores, n_samples=cfg.analysis.bootstrap_samples,
        confidence=cfg.analysis.confidence_level, seed=cfg.seed,
    )
    # Paired contrast: the same protein contributes its signal and decoy
    # enrichment to each resample, so protein-level variation cancels.
    paired = [s - d for s, d in zip(signal_scores, decoy_scores, strict=True)]
    excess_ci = bootstrap_enrichment(
        paired, n_samples=cfg.analysis.bootstrap_samples,
        confidence=cfg.analysis.confidence_level, seed=cfg.seed,
        family_size=n_layers * n_heads,
    )

    out = cfg.paths.analysis / f"{cfg.model.tag}_attention.npz"
    np.savez(
        out,
        signal_mean=signal_summary["mean"],
        signal_median=signal_summary["median"],
        signal_lo=signal_ci["lo"], signal_hi=signal_ci["hi"],
        decoy_mean=decoy_summary["mean"],
        decoy_median=decoy_summary["median"],
        decoy_lo=decoy_ci["lo"], decoy_hi=decoy_ci["hi"],
        excess_mean=excess_ci["mean"], excess_lo=excess_ci["lo"], excess_hi=excess_ci["hi"],
        best_head_per_layer=signal_summary["best_head_per_layer"],
        n_proteins=np.array(len(signal_scores)),
        n_decoy_draws=np.array(n_decoy_draws),
        n_unpaired=np.array(n_unpaired),
        excess_lo_adj=excess_ci["lo_adj"], excess_hi_adj=excess_ci["hi_adj"],
        # Per-protein arrays, so any further statistic (a different correction, a
        # per-compartment breakdown) can be computed without re-running the encoder.
        per_protein_signal=np.stack(signal_scores, axis=0).astype(np.float32),
        per_protein_decoy=np.stack(decoy_scores, axis=0).astype(np.float32),
    )
    logger.info(
        "Wrote attention enrichment %s (%d layers x %d heads, %d paired proteins, "
        "%d decoy draws; %d proteins dropped for lack of a downstream decoy window)",
        out, n_layers, n_heads, len(signal_scores), n_decoy_draws, n_unpaired,
    )
    return out


def run_bootstrap(cfg, embedding_path: Path, strategy: str, layer: int, val_fold: int) -> Path:
    """Refit the probe at one layer and bootstrap its test-set confidence interval."""
    from psl_probe.cli.run_probes import _align, _split_indices

    features_all, accessions = load_pooled(embedding_path, strategy)
    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")
    aligned = _align(dataset, accessions)
    train_index, val_index, test_index = _split_indices(aligned, cfg, val_fold)

    class_names = list(LOCALIZATION_CLASSES)
    labels = aligned[class_names].to_numpy(dtype=np.float32)
    features = features_all[:, layer, :]

    device = cfg.device if torch.cuda.is_available() else "cpu"
    result = train_probe(
        features[train_index], labels[train_index],
        features[val_index], labels[val_index],
        features[test_index], labels[test_index],
        class_names=class_names, cfg=cfg.probe, device=device,
        seed=cfg.seed, keep_state=True,
    )

    # Recover test scores from the retained probe for bootstrapping.
    from psl_probe.probes.linear import FeatureStandardizer, build_probe

    model = build_probe(cfg.probe, features.shape[1], len(class_names)).to(device)
    state = dict(result.state_dict or {})
    scaler_state = state.pop("_scaler", None)
    model.load_state_dict(state)
    x_test = torch.as_tensor(features[test_index], dtype=torch.float32, device=device)
    if scaler_state is not None:
        scaler = FeatureStandardizer().load_state_dict(
            {k: v.to(device) for k, v in scaler_state.items()}
        )
        x_test = scaler.transform(x_test)
    scores = sigmoid(predict_logits(model, x_test).cpu().numpy())

    point, lower, upper = bootstrap_ci(
        labels[test_index], scores, class_names,
        thresholds=result.thresholds,
        n_samples=cfg.analysis.bootstrap_samples,
        confidence=cfg.analysis.confidence_level,
        seed=cfg.seed,
    )
    # Consistency guard. This stage refits the probe from scratch, so if it is run
    # with different probe settings than run_probes used (a --set override, an
    # edited config), the confidence interval would silently describe a different
    # probe than the layer sweep it is reported alongside. Warn rather than fail:
    # a deliberate re-fit at different settings is legitimate, an accidental one
    # is a reporting error.
    results_path = cfg.paths.probes / f"{cfg.model.tag}_probe_results.csv"
    if results_path.exists():
        sweep = pd.read_csv(results_path)
        match = sweep[
            (sweep["task"] == "real")
            & (sweep["pooling"] == strategy)
            & (sweep["layer"] == layer)
        ]
        if not match.empty:
            recorded = float(match["macro_mcc"].iloc[0])
            if abs(recorded - point) > 0.02:
                logger.warning(
                    "Refit macro-MCC %.3f differs from the sweep's %.3f at layer %d. "
                    "The two stages were run with different probe settings "
                    "(compare results/logs/run_probes.resolved.yaml against "
                    "run_analysis.resolved.yaml); do not report them together.",
                    point, recorded, layer,
                )

    payload = {
        "model": cfg.model.tag, "pooling": strategy, "layer": layer,
        "metric": "macro_mcc", "point": point, "lower": lower, "upper": upper,
        "confidence": cfg.analysis.confidence_level,
        "n_bootstrap": cfg.analysis.bootstrap_samples,
        "per_class": result.per_class,
    }
    out = cfg.paths.analysis / f"{cfg.model.tag}_{strategy}_layer{layer}_bootstrap.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Bootstrap macro-MCC = %.3f [%.3f, %.3f]", point, lower, upper)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--strategy", default="mean", help="Pooling strategy to analyse")
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer for the bootstrap CI; defaults to the best layer in the probe results",
    )
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--skip-cka", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument(
        "--decoy-draws", type=int, default=5,
        help="Independent decoy placements per protein, averaged (default: %(default)s)",
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    cfg = resolve(args, "run_analysis")

    # A relative --embeddings is resolved against the repository root, matching
    # how every configured path behaves; otherwise it would silently resolve
    # against the working directory and fail from anywhere but the repo root.
    embedding_path = args.embeddings or (cfg.paths.embeddings / f"{cfg.model.tag}.h5")
    if not embedding_path.is_absolute():
        embedding_path = REPO_ROOT / embedding_path
    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")

    if cfg.analysis.run_cka and not args.skip_cka:
        run_cka(cfg, embedding_path, args.strategy)

    if cfg.analysis.run_attention and not args.skip_attention:
        run_attention(cfg, dataset, n_decoy_draws=args.decoy_draws)

    if not args.skip_bootstrap:
        layer = args.layer
        if layer is None:
            results_path = cfg.paths.probes / f"{cfg.model.tag}_probe_results.csv"
            if not results_path.exists():
                raise FileNotFoundError(
                    f"{results_path} not found; run run_probes first or pass --layer"
                )
            results = pd.read_csv(results_path)
            best = results[
                (results["task"] == "real") & (results["pooling"] == args.strategy)
            ].sort_values("macro_mcc", ascending=False).iloc[0]
            layer = int(best["layer"])
            logger.info("Selected best layer %d (macro-MCC %.3f)", layer, best["macro_mcc"])
        run_bootstrap(cfg, embedding_path, args.strategy, layer, args.val_fold)


if __name__ == "__main__":
    main()
