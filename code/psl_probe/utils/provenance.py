"""Capture of everything needed to re-create a run's context.

A metric is only reproducible if the conditions that produced it are recorded.
This module gathers those conditions --- code version, environment, hardware,
invocation --- into a plain dictionary that the tracker writes alongside every
run. Nothing here is specific to a particular experiment tracker.

The functions are deliberately failure-tolerant: a missing ``git`` binary or an
absent GPU degrades a field to ``None`` rather than aborting an experiment that
would otherwise succeed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def utc_now() -> str:
    """ISO-8601 UTC timestamp, used for run identifiers and timing fields."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    """Run a command and return its stripped stdout, or ``None`` on any failure."""
    try:
        out = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=30, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def git_state(repo: Path) -> dict[str, Any]:
    """Commit hash, branch and dirty-tree status.

    ``dirty`` is the field that matters for reproducibility: a commit hash alone
    is misleading when the working tree has uncommitted edits, so the list of
    modified files is recorded too.
    """
    if shutil.which("git") is None or not (repo / ".git").exists():
        return {"available": False}

    status = _run(["git", "status", "--porcelain"], cwd=repo)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)

    if status is None:
        # A failed or timed-out status must not be recorded as a clean tree: that
        # would assert reproducibility the run cannot support.
        logger.warning("git status unavailable; recording dirty state as unknown")
        return {
            "available": True,
            "commit": commit,
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo),
            "dirty": None,
            "dirty_files": None,
            "n_dirty_files": None,
            "note": "git status failed or timed out; working-tree state unknown",
        }

    dirty_files = [line[3:] for line in status.splitlines() if line.strip()]
    record = {
        "available": True,
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo),
        "dirty": bool(dirty_files),
        "dirty_files": sorted(dirty_files)[:50],
        "n_dirty_files": len(dirty_files),
    }
    if commit is None:
        record["note"] = (
            "repository has no commits yet; results cannot be tied to a code version "
            "until an initial commit exists"
        )
    return record


def package_versions() -> dict[str, str | None]:
    """Versions of the packages whose behaviour can change a numeric result."""
    import importlib.metadata as md

    watched = [
        "torch", "transformers", "numpy", "scipy", "scikit-learn", "pandas",
        "h5py", "biopython", "matplotlib", "pyarrow", "sentencepiece", "tokenizers",
    ]
    versions: dict[str, str | None] = {}
    for name in watched:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_state() -> dict[str, Any]:
    """CPU, RAM and CUDA device description."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # optional

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            first = meminfo.read_text().splitlines()[0]
            info["ram_gb"] = round(int(first.split()[1]) / 1e6, 1)

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
            info["driver"] = _run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            )
    except ImportError:
        info["cuda_available"] = None
    return info


def determinism_state() -> dict[str, Any]:
    """Settings that govern run-to-run numeric reproducibility.

    Recorded rather than asserted: some stages deliberately run non-deterministic
    kernels for throughput, and the report should say which.
    """
    state: dict[str, Any] = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    try:
        import torch

        state["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        state["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        state["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except ImportError:
        pass
    return state


def invocation() -> dict[str, Any]:
    """The exact command line, so a run can be replayed verbatim."""
    return {
        "argv": sys.argv,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "cwd": str(Path.cwd()),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "user": os.environ.get("USER"),
        "hostname": platform.node(),
    }


def collect(repo: Path) -> dict[str, Any]:
    """Assemble the full provenance record for a run."""
    return {
        "timestamp_utc": utc_now(),
        "invocation": invocation(),
        "git": git_state(repo),
        "packages": package_versions(),
        "hardware": hardware_state(),
        "determinism": determinism_state(),
    }


def sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, so multi-GB artifacts do not exhaust memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
