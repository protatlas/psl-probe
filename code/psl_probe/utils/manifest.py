"""Append-only manifest of produced artifacts.

``artifacts/manifest.json`` answers one question: for any file the pipeline
produced, which run produced it, with which command, from which code version, and
does the file on disk still match. It is append-only --- re-generating a file adds
a new entry rather than replacing the old one, so a changed hash is visible as a
history rather than silently overwritten.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psl_probe.utils.logging import get_logger
from psl_probe.utils.provenance import sha256

logger = get_logger(__name__)

MANIFEST_VERSION = 1


def manifest_path(repo_root: Path) -> Path:
    return Path(repo_root) / "artifacts" / "manifest.json"


def load_manifest(repo_root: Path) -> dict[str, Any]:
    """Read the manifest, returning an empty one if it does not exist yet."""
    path = manifest_path(repo_root)
    if not path.exists():
        return {"version": MANIFEST_VERSION, "entries": []}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def append_entries(repo_root: Path, entries: list[dict[str, Any]]) -> Path:
    """Append artifact entries and write the manifest back."""
    if not entries:
        return manifest_path(repo_root)
    manifest = load_manifest(repo_root)
    manifest["entries"].extend(entries)
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = manifest_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Manifest: +%d entries (%d total) -> %s",
                len(entries), len(manifest["entries"]), path)
    return path


def verify(repo_root: Path, recompute: bool = True,
           latest_only: bool = True) -> list[dict[str, Any]]:
    """Check every manifest entry against the file on disk.

    Returns one status record per entry with ``status`` in:

    ``ok``
        The file exists and its hash matches the recorded one.
    ``missing``
        The file has been deleted or moved.
    ``changed``
        The file exists but its content no longer matches --- a result in the
        paper attributed to that artifact can no longer be traced to it.
    ``unhashed``
        The entry was registered without a hash, so only existence is checked.

    Because the manifest is append-only, a regenerated file has several entries.
    By default only the most recent entry per path is checked --- that is the
    question worth asking ("does the current state match what was last
    recorded?"). Pass ``latest_only=False`` to audit the full history, where
    older entries are expected to read as ``changed``.
    """
    repo_root = Path(repo_root)
    entries = load_manifest(repo_root).get("entries", [])
    if latest_only:
        by_path: dict[str, dict[str, Any]] = {}
        for entry in entries:
            by_path[entry["path"]] = entry   # later entries win
        entries = list(by_path.values())

    results = []
    for entry in entries:
        path = repo_root / entry["path"]
        record = {"path": entry["path"], "run_id": entry.get("run_id"),
                  "stage": entry.get("stage")}
        if not path.exists():
            record["status"] = "missing"
        elif entry.get("sha256") is None:
            record["status"] = "unhashed"
        elif not recompute:
            record["status"] = "ok"
        else:
            record["status"] = "ok" if sha256(path) == entry["sha256"] else "changed"
        results.append(record)
    return results


def latest_for_path(repo_root: Path, relative_path: str) -> dict[str, Any] | None:
    """Most recent manifest entry for a given artifact path."""
    matches = [e for e in load_manifest(repo_root).get("entries", [])
               if e["path"] == str(relative_path)]
    return matches[-1] if matches else None
