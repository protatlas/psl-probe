"""Stage 2: extract layer-wise pLM representations for the benchmark.

Example
-------
::

    python -m psl_probe.cli.extract_embeddings --config configs/esm2_650m.yaml
    python -m psl_probe.cli.extract_embeddings --ablation nterm_delete --ablation-n 40
"""

from __future__ import annotations

import argparse

import torch

from psl_probe.analysis.ablation import MODES, build_ablated_dataset
from psl_probe.cli import add_common_arguments, resolve
from psl_probe.data.curate import load_dataset
from psl_probe.embeddings.extract import extract_embeddings
from psl_probe.embeddings.models import load_plm
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--ablation", choices=MODES, default="none", help="Sequence ablation mode")
    parser.add_argument("--ablation-n", type=int, default=0, help="Residues affected by ablation")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing embeddings")
    args = parser.parse_args()
    cfg = resolve(args, f"extract_{args.__dict__.get('ablation', 'none')}")

    if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
        logger.warning("CUDA unavailable; falling back to CPU (this will be slow)")
        cfg.device = "cpu"

    dataset = load_dataset(cfg.paths.processed / "benchmark.parquet")
    suffix = ""
    if args.ablation != "none" and args.ablation_n > 0:
        dataset = build_ablated_dataset(dataset, args.ablation, args.ablation_n, seed=cfg.seed)
        suffix = f".{args.ablation}{args.ablation_n}"

    plm = load_plm(cfg.model, device=cfg.device)
    output = cfg.paths.embeddings / f"{cfg.model.tag}{suffix}.h5"
    extract_embeddings(dataset, plm, cfg, output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
