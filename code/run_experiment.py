#!/usr/bin/env python3
"""Single tracked entry point for every experiment in the paper.

All results reported in the manuscript are produced through this script, so that
each has a run directory recording the exact command, code version, environment,
seeds, configuration, timings and output hashes::

    python code/run_experiment.py --config configs/experiments/main_650m.yaml

An experiment config lists the pipeline stages to run and the model config they
apply to. Stages are the same ones exposed as individual CLIs in
``psl_probe.cli``; this script adds tracking, artifact registration and ordering.
Running a stage directly still works and is useful for debugging, but produces no
run record, so anything destined for the paper should go through here.

Determinism is on by default (``--deterministic``); see ``REPRODUCIBILITY.md``
for the sources of variability that cannot be eliminated.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from psl_probe.config import Config  # noqa: E402
from psl_probe.tracking import ExperimentTracker  # noqa: E402
from psl_probe.utils.io import load_yaml, save_yaml  # noqa: E402
from psl_probe.utils.logging import get_logger, setup_logging  # noqa: E402
from psl_probe.utils.manifest import append_entries  # noqa: E402

logger = get_logger(__name__)

#: Stage name -> CLI module. Order here is the canonical pipeline order.
STAGES: dict[str, str] = {
    "prepare_data": "psl_probe.cli.prepare_data",
    "extract_embeddings": "psl_probe.cli.extract_embeddings",
    "run_probes": "psl_probe.cli.run_probes",
    "run_composition_baseline": "psl_probe.cli.run_composition_baseline",
    "run_analysis": "psl_probe.cli.run_analysis",
    "run_ablation": "psl_probe.cli.run_ablation",
    "run_pooling_contrast": "psl_probe.cli.run_pooling_contrast",
    "make_split_variants": "psl_probe.cli.make_split_variants",
    "make_figures": "psl_probe.cli.make_figures",
    "make_tables": "psl_probe.cli.make_tables",
}

#: Which configured directory each stage writes to, and which file patterns to
#: register from it. Resolved against the run's own config rather than hard-coded
#: globs: a wildcard such as ``results/embeddings*/`` would also descend into
#: unrelated (and possibly unreadable) sibling directories.
STAGE_OUTPUTS: dict[str, tuple[str, list[str]]] = {
    "prepare_data": ("processed", ["*.parquet", "*.csv"]),
    "extract_embeddings": ("embeddings", ["*.h5"]),
    "run_probes": ("probes", ["*.csv"]),
    "run_composition_baseline": ("probes", ["composition_baseline*.csv"]),
    "run_analysis": ("analysis", ["*.npy", "*.npz", "*.json"]),
    "run_ablation": ("analysis", ["*ablation*.csv"]),
    "run_pooling_contrast": ("analysis", ["*pooling*.csv"]),
    "make_figures": ("figures", ["*.pdf"]),
}

#: Stages whose outputs are not under a single configured path.
STAGE_OUTPUT_DIRS: dict[str, list[str]] = {
    "make_tables": ["latex/tables"],
    "make_split_variants": ["datasets/processed_id50", "datasets/processed_id90",
                            "datasets/processed_random"],
}

#: Artifacts above this size are registered by size and mtime rather than hashed;
#: hashing the full embedding set costs minutes and buys little over mtime here.
HASH_SIZE_LIMIT_BYTES = 2 * 1024**3

#: How often to sample a running stage's GPU memory, in seconds. Short enough
#: to catch an extraction peak, long enough not to add measurable overhead.
GPU_SAMPLE_INTERVAL_S = 2.0


def build_stage_command(stage: str, model_config: Path, args: list[str],
                        overrides: list[str]) -> list[str]:
    """Assemble the argv for one stage."""
    command = [sys.executable, "-m", STAGES[stage], "--config", str(model_config)]
    command += list(args)
    for override in overrides:
        command += ["--set", override]
    return command


def _gpu_memory_for_tree(root_pid: int) -> float:
    """Summed GPU memory (GB) of ``root_pid`` and its descendants, via nvidia-smi.

    Stages run as subprocesses, so ``torch.cuda.max_memory_allocated`` in the
    orchestrator is always zero. Sampling the driver's per-process accounting is
    the only way to record a stage's true peak from outside it.
    """
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return 0.0

    try:
        children = subprocess.run(
            ["pgrep", "-P", str(root_pid)], capture_output=True, text=True, timeout=5
        ).stdout.split()
    except (subprocess.SubprocessError, OSError):
        children = []
    tree = {str(root_pid), *children}

    total_mib = 0.0
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] in tree:
            try:
                total_mib += float(parts[1])
            except ValueError:
                pass
    return total_mib / 1024.0


def run_stage(stage: str, command: list[str], tracker: ExperimentTracker,
              cwd: Path) -> dict[str, Any]:
    """Execute one stage, streaming its output and sampling its GPU footprint."""
    console = tracker.run_dir / "console.log"
    started = time.time()
    logger.info("[%s] %s", stage, " ".join(command))

    peak_gpu = 0.0
    with open(console, "a", encoding="utf-8") as handle:
        handle.write(f"\n{'='*70}\n$ {' '.join(command)}\n{'='*70}\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=cwd, stdout=handle,
                                   stderr=subprocess.STDOUT)
        while process.poll() is None:
            peak_gpu = max(peak_gpu, _gpu_memory_for_tree(process.pid))
            time.sleep(GPU_SAMPLE_INTERVAL_S)
        completed = process

    elapsed = time.time() - started
    record = {"stage": stage, "returncode": completed.returncode,
              "wall_clock_s": round(elapsed, 2),
              "gpu_peak_gb": round(peak_gpu, 2), "command": " ".join(command)}
    tracker.log_metrics(record, context={"kind": "stage"})
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage {stage!r} failed with exit code {completed.returncode}. "
            f"See {console} for output."
        )
    logger.info("[%s] completed in %.1f s", stage, elapsed)
    return record


def stage_output_dirs(stage: str, resolved: Config) -> list[tuple[Path, list[str]]]:
    """Directories and file patterns to scan for a stage's outputs."""
    targets: list[tuple[Path, list[str]]] = []
    if stage in STAGE_OUTPUTS:
        attr, patterns = STAGE_OUTPUTS[stage]
        targets.append((Path(getattr(resolved.paths, attr)), patterns))
    for relative in STAGE_OUTPUT_DIRS.get(stage, []):
        suffix = ["*.tex"] if relative.startswith("latex") else ["*.parquet", "*.csv"]
        targets.append((REPO_ROOT / relative, suffix))
    return targets


def register_outputs(stage: str, tracker: ExperimentTracker, since: float,
                     resolved: Config) -> int:
    """Register files this stage produced or updated.

    Only the directories the stage's own configuration points at are scanned, so
    an unrelated or unreadable sibling directory cannot stall the run.
    """
    registered = 0
    for directory, patterns in stage_output_dirs(stage, resolved):
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                try:
                    stat = path.stat()
                except OSError as exc:  # unreadable file: note it, do not abort
                    logger.warning("Could not stat %s: %s", path, exc)
                    continue
                if stat.st_mtime < since:
                    continue  # untouched by this stage
                entry = tracker.log_artifact(
                    path, kind=stage,
                    description=f"produced by stage {stage}",
                    hash_it=stat.st_size <= HASH_SIZE_LIMIT_BYTES,
                )
                registered += 1 if entry else 0
    return registered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True,
                        help="Experiment config (see configs/experiments/)")
    parser.add_argument("--stages", nargs="+", default=None,
                        help="Override the stage list from the config")
    parser.add_argument("--tracker", default="auto",
                        choices=["auto", "wandb", "mlflow", "local"])
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Enable deterministic kernels (default: on)")
    parser.add_argument("--no-deterministic", dest="deterministic",
                        action="store_false",
                        help="Allow non-deterministic kernels for throughput")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the stage commands without executing them")
    parser.add_argument("--notes", default=None, help="Free-text note stored with the run")
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    name = experiment.get("name", args.config.stem)
    model_config = REPO_ROOT / experiment["model_config"]
    stages = args.stages or experiment["stages"]
    stage_args: dict[str, list[str]] = experiment.get("stage_args", {})
    overrides: list[str] = experiment.get("overrides", [])

    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; expected some of {sorted(STAGES)}")

    setup_logging("INFO")
    resolved = Config.from_yaml(model_config).apply_overrides(overrides)

    tracked_config = {
        "experiment": name,
        "experiment_config": str(args.config.relative_to(REPO_ROOT))
        if args.config.is_absolute() else str(args.config),
        "model_config": str(model_config.relative_to(REPO_ROOT)),
        "stages": stages,
        "overrides": overrides,
        "deterministic": args.deterministic,
        "resolved": resolved.to_dict(),
    }

    def stage_extra(stage: str) -> list[str]:
        """Per-stage CLI arguments, including the determinism flag where supported."""
        extra = list(stage_args.get(stage, []))
        if args.deterministic and stage in {"run_probes", "run_composition_baseline",
                                            "run_ablation", "run_pooling_contrast"}:
            extra.append("--deterministic")
        return extra

    if args.dry_run:
        for stage in stages:
            print(" ".join(build_stage_command(
                stage, model_config, stage_extra(stage), overrides)))
        return

    with ExperimentTracker(
        stage=name, config=tracked_config, repo_root=REPO_ROOT,
        backend=args.tracker, tags=[*stages, resolved.model.tag], notes=args.notes,
    ) as tracker:
        save_yaml(resolved.to_dict(), tracker.run_dir / "config.resolved.yaml")
        (tracker.run_dir / "experiment.yaml").write_text(
            args.config.read_text(), encoding="utf-8")

        durations: dict[str, float] = {}
        gpu_peaks: dict[str, float] = {}
        for stage in stages:
            started = time.time()
            record = run_stage(
                stage,
                build_stage_command(stage, model_config, stage_extra(stage), overrides),
                tracker, cwd=REPO_ROOT / "code",
            )
            durations[stage] = record["wall_clock_s"]
            gpu_peaks[stage] = record["gpu_peak_gb"]
            register_outputs(stage, tracker, since=started, resolved=resolved)

        tracker.log_summary({
            "n_stages": len(stages),
            "total_stage_seconds": round(sum(durations.values()), 2),
            "n_artifacts": len(tracker.artifacts),
            "max_stage_gpu_peak_gb": max(gpu_peaks.values(), default=0.0),
            **{f"seconds/{k}": v for k, v in durations.items()},
            **{f"gpu_peak_gb/{k}": v for k, v in gpu_peaks.items()},
        })
        append_entries(REPO_ROOT, tracker.artifacts)

    print(f"\nRun complete. Logs and provenance: logs/{tracker.run_id}/")


if __name__ == "__main__":
    main()
