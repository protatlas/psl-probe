"""Project-wide logging configuration.

A single :func:`setup_logging` call at CLI entry configures the root logger; every
module then obtains a child logger via :func:`get_logger(__name__)`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Configure the root logger for console and (optionally) file output.

    Parameters
    ----------
    level:
        Logging level, either a :mod:`logging` constant or its name.
    log_file:
        If given, log records are additionally appended to this file. Parent
        directories are created on demand.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    root = logging.getLogger()
    root.setLevel(level)
    # Reset handlers so repeated calls (e.g. in notebooks) do not duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Third-party libraries are chatty at INFO; keep them at WARNING.
    for noisy in ("transformers", "urllib3", "matplotlib", "numba", "h5py"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
