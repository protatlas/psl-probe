"""Stage 4c: contrast two pooling operators at a fixed layer.

This is the inferential counterpart to the \\nterm{}-retention claim: how much of
a compartment's decodability survives when the pooled representation is restricted
to the first ``w`` residues instead of the whole protein. As with the ablation
contrasts, both probes are scored on the same test proteins (so the bootstrap is
paired), each is refitted under several seeds (so the interval includes
probe-fitting variance), and significance across the ten compartments is
controlled with Benjamini-Hochberg q-values.

Both the difference and the retention *ratio* are reported. The ratio is
bootstrapped on shared resamples rather than formed from two independently
bootstrapped quantities, which would ignore the correlation between numerator and
denominator.

Example
-------
::

    python -m psl_probe.cli.run_pooling_contrast --config configs/esm2_650m.yaml \\
        --layer 33 --reference mean --comparison nterm --seeds 5
"""

from __future__ import annotations

import argparse

import pandas as pd

from psl_probe.analysis.metrics import multiseed_paired_bootstrap_deltas
from psl_probe.cli import add_common_arguments, resolve
from psl_probe.cli.run_ablation import _fit
from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--layer", type=int, required=True, help="Layer to probe")
    parser.add_argument("--reference", default="mean", help="Reference pooling strategy")
    parser.add_argument("--comparison", default="nterm", help="Pooling strategy to contrast")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument(
        "--seeds", type=int, default=5, metavar="N",
        help="Refit both probes under N seeds and pool their bootstrap draws.",
    )
    args = parser.parse_args()
    cfg = resolve(args, "run_pooling_contrast")

    embedding_path = cfg.paths.embeddings / f"{cfg.model.tag}.h5"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"{embedding_path} not found; run extract_embeddings before this stage."
        )

    seeds = [cfg.seed + i for i in range(max(args.seeds, 1))]
    logger.info(
        "Contrasting pooling '%s' against '%s' at layer %d under %d seed(s)",
        args.comparison, args.reference, args.layer, len(seeds),
    )

    reference = _fit(cfg, embedding_path, args.reference, args.layer, args.val_fold, seeds)
    comparison = _fit(cfg, embedding_path, args.comparison, args.layer, args.val_fold, seeds)

    records = multiseed_paired_bootstrap_deltas(
        reference[0].test_labels,
        [r.test_scores for r in reference],
        [c.test_scores for c in comparison],
        [r.thresholds for r in reference],
        [c.thresholds for c in comparison],
        class_names=list(LOCALIZATION_CLASSES),
        n_samples_per_seed=cfg.analysis.bootstrap_samples,
        confidence=cfg.analysis.confidence_level,
        seed=cfg.seed,
        include_ratio=True,
    )

    frame = pd.DataFrame(records)
    frame["reference_pooling"] = args.reference
    frame["comparison_pooling"] = args.comparison
    frame["layer"] = args.layer

    out = (
        cfg.paths.analysis
        / f"{cfg.model.tag}_pooling_{args.comparison}_vs_{args.reference}_layer{args.layer}.csv"
    )
    frame.to_csv(out, index=False)

    for row in sorted(records, key=lambda r: r["ratio"], reverse=True):
        logger.info(
            "%-22s %s=%.3f %s=%.3f  retention %.0f%% [%.0f%%, %.0f%%]  delta %+.3f  q=%.3g",
            row["class"], args.reference, row["reference_mcc"],
            args.comparison, row["comparison_mcc"],
            100 * row["ratio"], 100 * row["ratio_lo"], 100 * row["ratio_hi"],
            row["delta"], row["q_bh"],
        )
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
