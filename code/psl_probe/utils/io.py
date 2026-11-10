"""Small, typed I/O helpers used across the pipeline."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml


def _open_maybe_gzip(path: Path, mode: str = "rt"):
    """Open ``path`` transparently, decompressing when the suffix is ``.gz``."""
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8" if "t" in mode else None)
    return open(path, mode, encoding="utf-8" if "t" in mode else None)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML document into a dictionary."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(obj: dict[str, Any], path: Path | str) -> None:
    """Write a dictionary to YAML, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False, default_flow_style=False)


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records from a (optionally gzipped) JSON Lines file."""
    path = Path(path)
    with _open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path | str) -> int:
    """Write records as JSON Lines; returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_maybe_gzip(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_fasta(records: Iterable[tuple[str, str]], path: Path | str, width: int = 60) -> int:
    """Write ``(identifier, sequence)`` pairs to a FASTA file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), width):
                handle.write(sequence[start : start + width] + "\n")
            count += 1
    return count


def read_fasta(path: Path | str) -> Iterator[tuple[str, str]]:
    """Stream ``(identifier, sequence)`` pairs from a FASTA file.

    The identifier is the full header line minus the leading ``>``.
    """
    path = Path(path)
    header: str | None = None
    chunks: list[str] = []
    with _open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            elif line:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)
