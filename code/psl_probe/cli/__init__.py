"""Command-line entry points.

Every stage shares the same argument surface via :func:`add_common_arguments`, so
a run is fully described by ``--config`` plus ``--set`` overrides; the resolved
configuration is written next to the outputs by :func:`resolve`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from psl_probe.config import REPO_ROOT, Config
from psl_probe.utils.io import save_yaml
from psl_probe.utils.logging import get_logger, setup_logging
from psl_probe.utils.seed import seed_everything

logger = get_logger(__name__)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "esm2_650m.yaml"


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach ``--config``, ``--set``, ``--log-level`` and ``--deterministic``."""
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="YAML configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE", default=[],
        help="Dotted config override, repeatable (e.g. --set probe.epochs=10)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: %(default)s)")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Enable deterministic GPU kernels (slower, bit-reproducible)",
    )
    return parser


def resolve(args: argparse.Namespace, run_name: str) -> Config:
    """Load the config, apply overrides, set up logging/seeding and record provenance."""
    cfg = Config.from_yaml(args.config).apply_overrides(args.overrides)
    cfg.paths.mkdirs()
    setup_logging(args.log_level, cfg.paths.logs / f"{run_name}.log")
    seed_everything(cfg.seed, deterministic=getattr(args, "deterministic", False))
    save_yaml(cfg.to_dict(), cfg.paths.logs / f"{run_name}.resolved.yaml")
    logger.info("Resolved config written to %s", cfg.paths.logs / f"{run_name}.resolved.yaml")
    return cfg
