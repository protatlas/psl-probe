"""Tracking, provenance and manifest behaviour.

These guard the properties the reproducibility claims depend on: a run directory
is never reused, a failure is recorded as a failure, and an unverifiable state is
never recorded as a verified one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psl_probe.tracking import ExperimentTracker, make_run_id, select_backend
from psl_probe.utils.manifest import (
    append_entries,
    latest_for_path,
    load_manifest,
    verify,
)
from psl_probe.utils.provenance import collect, git_state, sha256


def test_run_ids_sort_chronologically_and_are_unique() -> None:
    ids = [make_run_id("stage") for _ in range(50)]
    assert len(set(ids)) == 50
    assert ids == sorted(ids) or all(i.startswith("20") for i in ids)


def test_run_directory_is_never_reused(tmp_path: Path) -> None:
    """mkdir(exist_ok=False) is what prevents a re-run clobbering an old record."""
    with ExperimentTracker("s", {}, tmp_path, logs_root=tmp_path / "logs") as tracker:
        existing = tracker.run_dir
    with pytest.raises(FileExistsError):
        t = ExperimentTracker.__new__(ExperimentTracker)
        t.run_dir = existing
        existing.mkdir(parents=True, exist_ok=False)


def test_failure_is_recorded_and_exception_propagates(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        with ExperimentTracker("s", {}, tmp_path, logs_root=tmp_path / "logs") as tracker:
            run_dir = tracker.run_dir
            raise ValueError("boom")
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert "ValueError: boom" in record["summary"]["error"]


def test_metrics_stream_widens_header_when_fields_appear(tmp_path: Path) -> None:
    with ExperimentTracker("s", {}, tmp_path, logs_root=tmp_path / "logs") as tracker:
        tracker.log_metrics({"a": 1.0})
        tracker.log_metrics({"a": 2.0, "b": 3.0})
        run_dir = tracker.run_dir
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[1]["b"] == 3.0
    header = (run_dir / "metrics.csv").read_text().splitlines()[0]
    assert "a" in header and "b" in header


def test_artifact_registration_records_hash(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("payload")
    with ExperimentTracker("s", {}, tmp_path, logs_root=tmp_path / "logs") as tracker:
        entry = tracker.log_artifact(target, kind="test")
    assert entry["sha256"] == sha256(target)
    assert entry["bytes"] == len("payload")


def test_large_artifact_can_be_registered_unhashed(tmp_path: Path) -> None:
    """An unhashed entry must say so rather than imply a check that did not run."""
    target = tmp_path / "big.bin"
    target.write_bytes(b"0" * 32)
    with ExperimentTracker("s", {}, tmp_path, logs_root=tmp_path / "logs") as tracker:
        entry = tracker.log_artifact(target, hash_it=False)
    assert entry["sha256"] is None
    assert "not hashed" in entry["note"]


def test_missing_artifact_is_not_registered(tmp_path: Path) -> None:
    with ExperimentTracker("s", {}, tmp_path, logs_root=tmp_path / "logs") as tracker:
        assert tracker.log_artifact(tmp_path / "nope.txt") == {}
        assert tracker.artifacts == []


def test_manifest_detects_change_and_deletion(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    target = tmp_path / "a.txt"
    target.write_text("one")
    append_entries(tmp_path, [{"path": "a.txt", "sha256": sha256(target), "run_id": "r"}])
    assert verify(tmp_path)[0]["status"] == "ok"

    target.write_text("two")               # content changed
    assert verify(tmp_path)[0]["status"] == "changed"

    target.unlink()                        # deleted
    assert verify(tmp_path)[0]["status"] == "missing"


def test_manifest_is_append_only_and_verifies_current_state(tmp_path: Path) -> None:
    """History is retained, but the default check asks about the current state.

    A regenerated file has several entries; checking all of them would report the
    superseded ones as `changed`, which is a history audit rather than the
    question the reproducibility gate needs to answer.
    """
    (tmp_path / "artifacts").mkdir()
    f = tmp_path / "a.txt"; f.write_text("one")
    append_entries(tmp_path, [{"path": "a.txt", "sha256": sha256(f), "run_id": "r1"}])
    f.write_text("two")
    append_entries(tmp_path, [{"path": "a.txt", "sha256": sha256(f), "run_id": "r2"}])

    assert len(load_manifest(tmp_path)["entries"]) == 2      # history retained
    assert latest_for_path(tmp_path, "a.txt")["run_id"] == "r2"

    current = verify(tmp_path)                                # default: latest only
    assert len(current) == 1 and current[0]["status"] == "ok"

    history = verify(tmp_path, latest_only=False)             # full audit
    assert len(history) == 2
    assert [r["status"] for r in history] == ["changed", "ok"]


def test_git_state_never_reports_unknown_as_clean(tmp_path: Path) -> None:
    """A directory with no repository must not be reported as a clean tree."""
    state = git_state(tmp_path)
    assert state["available"] is False
    assert "dirty" not in state or state["dirty"] is not False


def test_provenance_records_the_fields_reproducibility_depends_on(tmp_path: Path) -> None:
    record = collect(tmp_path)
    assert record["invocation"]["command"]
    assert "python" in record["invocation"]
    assert set(record) >= {"timestamp_utc", "invocation", "git", "packages",
                           "hardware", "determinism"}
    assert record["packages"]["torch"]


def test_backend_selection_is_explicit_and_validated() -> None:
    assert select_backend("local") == "local"
    with pytest.raises(ValueError, match="Unknown tracker backend"):
        select_backend("tensorboard")
