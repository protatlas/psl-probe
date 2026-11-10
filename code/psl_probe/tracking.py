"""Structured experiment tracking with graceful backend fallback.

Every run gets a directory ``logs/<run_id>/`` containing its provenance record,
resolved configuration, metric stream and a manifest of the files it produced.
That directory is written regardless of which backend is active, so results stay
auditable on a machine with no tracking service configured.

Backend selection, highest preference first:

``wandb``
    Used when the package is importable *and* credentials are present
    (``WANDB_API_KEY``, or a netrc entry). Offline mode is used when
    ``WANDB_MODE=offline`` is set.
``mlflow``
    Used when importable. Logs to ``logs/mlruns`` by default.
``local``
    Always available. JSON + CSV files under ``logs/<run_id>/``.

The local backend is not a degraded mode: it records the same fields as the
others, and is the format the reproducibility checks in ``scripts/`` read.
"""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from psl_probe.utils.logging import get_logger
from psl_probe.utils.provenance import collect, sha256, utc_now

logger = get_logger(__name__)

#: Backends in descending order of preference.
BACKENDS = ("wandb", "mlflow", "local")


def _wandb_available() -> bool:
    """True when wandb is importable and has credentials to use."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("WANDB_MODE") == "offline":
        return True
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    return netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore")


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def select_backend(preference: str = "auto") -> str:
    """Resolve the backend to use, honouring an explicit preference."""
    if preference != "auto":
        if preference not in BACKENDS:
            raise ValueError(f"Unknown tracker backend {preference!r}; expected {BACKENDS}")
        return preference
    if _wandb_available():
        return "wandb"
    if _mlflow_available():
        return "mlflow"
    return "local"


def make_run_id(stage: str) -> str:
    """Timestamped, collision-resistant run identifier.

    Sorting lexicographically orders runs chronologically, which matters because
    run directories are never overwritten.
    """
    return f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{stage}_{uuid.uuid4().hex[:6]}"


class ExperimentTracker:
    """Records one run: provenance, config, metrics, artifacts and timing.

    Use as a context manager so that wall-clock time, peak GPU memory and the
    final status are recorded even when the run raises::

        with ExperimentTracker("run_probes", cfg, repo_root) as tracker:
            tracker.log_metrics({"macro_mcc": 0.558}, step=33)
            tracker.log_artifact(path)
    """

    def __init__(
        self,
        stage: str,
        config: dict[str, Any],
        repo_root: Path,
        logs_root: Path | None = None,
        backend: str = "auto",
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        self.stage = stage
        self.config = config
        self.repo_root = Path(repo_root)
        self.logs_root = Path(logs_root or self.repo_root / "logs")
        self.run_id = make_run_id(stage)
        self.run_dir = self.logs_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run
        self.backend = select_backend(backend)
        self.tags = tags or []
        self.notes = notes

        self.provenance = collect(self.repo_root)
        self._metrics_path = self.run_dir / "metrics.csv"
        self._metrics_jsonl = self.run_dir / "metrics.jsonl"
        self._metric_fields: list[str] = []
        self._artifacts: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}
        self._start = time.time()
        self._handle: Any = None

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self) -> ExperimentTracker:
        self._reset_gpu_peak()
        self._open_backend()
        self._write_run_record(status="running")
        logger.info(
            "Run %s | stage=%s | backend=%s | logs=%s",
            self.run_id, self.stage, self.backend, self.run_dir,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "failed" if exc_type is not None else "completed"
        if exc_type is not None:
            self._summary["error"] = f"{exc_type.__name__}: {exc}"
        self._write_run_record(status=status)
        self._close_backend(status)
        logger.info(
            "Run %s %s in %.1f s | peak GPU %.2f GB",
            self.run_id, status, time.time() - self._start,
            self._gpu_peak_gb() or 0.0,
        )
        return False  # never suppress the exception

    # -- backends -----------------------------------------------------------
    def _open_backend(self) -> None:
        if self.backend == "wandb":
            import wandb

            self._handle = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "psl-probe"),
                name=self.run_id,
                config={**self.config, "_provenance": self.provenance},
                tags=self.tags,
                notes=self.notes,
                dir=str(self.logs_root),
            )
        elif self.backend == "mlflow":
            import mlflow

            mlflow.set_tracking_uri((self.logs_root / "mlruns").as_uri())
            mlflow.set_experiment("psl-probe")
            self._handle = mlflow.start_run(run_name=self.run_id)
            mlflow.log_params(_flatten(self.config))
            mlflow.set_tags({"stage": self.stage, **{t: "1" for t in self.tags}})

    def _close_backend(self, status: str) -> None:
        if self.backend == "wandb" and self._handle is not None:
            import wandb

            wandb.summary.update(self._summary)
            wandb.finish(exit_code=0 if status == "completed" else 1)
        elif self.backend == "mlflow" and self._handle is not None:
            import mlflow

            for key, value in self._summary.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"summary/{key}", value)
            mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")

    # -- logging ------------------------------------------------------------
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None,
                    context: dict[str, Any] | None = None) -> None:
        """Append a metric record to the local stream and the active backend."""
        record = {"_time": utc_now(), "_step": step, **(context or {}), **metrics}

        with open(self._metrics_jsonl, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

        # CSV needs a stable header; widen it if a later record adds fields.
        new_fields = [k for k in record if k not in self._metric_fields]
        if new_fields:
            self._metric_fields.extend(new_fields)
            self._rewrite_csv_header()
        with open(self._metrics_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._metric_fields).writerow(
                {k: record.get(k, "") for k in self._metric_fields}
            )

        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if self.backend == "wandb" and numeric:
            import wandb

            wandb.log(numeric, step=step)
        elif self.backend == "mlflow" and numeric:
            import mlflow

            for key, value in numeric.items():
                mlflow.log_metric(key, value, step=step or 0)

    def _rewrite_csv_header(self) -> None:
        """Rewrite the CSV in place when the field set grows mid-run."""
        rows = list(self._read_jsonl())
        with open(self._metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._metric_fields)
            writer.writeheader()
            for row in rows[:-1] if rows else []:
                writer.writerow({k: row.get(k, "") for k in self._metric_fields})

    def _read_jsonl(self) -> list[dict[str, Any]]:
        if not self._metrics_jsonl.exists():
            return []
        with open(self._metrics_jsonl, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Record headline values for the run (best metric, counts, and so on)."""
        self._summary.update(summary)

    def log_artifact(self, path: Path | str, kind: str = "output",
                     description: str | None = None, hash_it: bool = True) -> dict[str, Any]:
        """Register a produced file with its SHA-256 and the command that made it.

        Large files can be registered with ``hash_it=False``; the entry then
        records size and mtime only, and says so, rather than implying a
        verification that was not performed.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Artifact %s does not exist; not registered", path)
            return {}
        entry: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": self.stage,
            "path": str(path.relative_to(self.repo_root)) if path.is_relative_to(self.repo_root)
                    else str(path),
            "kind": kind,
            "description": description,
            "bytes": path.stat().st_size,
            "modified_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)
            ),
            "command": self.provenance["invocation"]["command"],
            "git_commit": self.provenance["git"].get("commit"),
        }
        if hash_it:
            entry["sha256"] = sha256(path)
        else:
            entry["sha256"] = None
            entry["note"] = "not hashed (registered by size and mtime only)"
        self._artifacts.append(entry)
        return entry

    # -- GPU ----------------------------------------------------------------
    @staticmethod
    def _reset_gpu_peak() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    @staticmethod
    def _gpu_peak_gb() -> float | None:
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / 1e9, 3)
        except ImportError:
            pass
        return None

    # -- persistence --------------------------------------------------------
    def _write_run_record(self, status: str) -> None:
        record = {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": status,
            "backend": self.backend,
            "tags": self.tags,
            "notes": self.notes,
            "started_utc": self.provenance["timestamp_utc"],
            "ended_utc": utc_now() if status != "running" else None,
            "wall_clock_s": round(time.time() - self._start, 2),
            "gpu_peak_allocated_gb": self._gpu_peak_gb(),
            "gpu_peak_reserved_gb": self._gpu_peak_reserved_gb(),
            "config": self.config,
            "provenance": self.provenance,
            "summary": self._summary,
            "artifacts": self._artifacts,
        }
        (self.run_dir / "run.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def _gpu_peak_reserved_gb() -> float | None:
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_reserved() / 1e9, 3)
        except ImportError:
            pass
        return None

    @property
    def artifacts(self) -> list[dict[str, Any]]:
        return list(self._artifacts)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config into dotted keys, for trackers that want scalars."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, (list, tuple)):
        flat[prefix.rstrip(".")] = json.dumps(obj, default=str)
    else:
        flat[prefix.rstrip(".")] = obj
    return flat
