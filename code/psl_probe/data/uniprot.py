"""Swiss-Prot acquisition and parsing.

The benchmark is built from the reviewed (Swiss-Prot) section of UniProtKB, which
is the only public resource that provides *both* curated subcellular-location
annotations with explicit evidence codes *and* curated sorting-signal sequence
features (``SIGNAL``, ``TRANSIT``, ``TRANSMEM``). Both are required by this study:
the former supplies probe targets, the latter supplies the residue-level ground
truth against which we test whether pLM representations localise information to
sorting signals.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

from psl_probe.data.labels import (
    SIGNAL_FEATURE_TYPES,
    map_subcellular_location,
)
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

SWISSPROT_URL_TEMPLATE = (
    "https://ftp.uniprot.org/pub/databases/uniprot/{release}/knowledgebase/"
    "complete/uniprot_sprot.dat.gz"
)

RELEASE_NOTE_URL_TEMPLATE = (
    "https://ftp.uniprot.org/pub/databases/uniprot/{release}/relnotes.txt"
)

#: The 20 standard amino acids plus the ambiguity codes pLM tokenizers accept.
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_RESIDUES = set("BJOUXZ")


@dataclass
class SignalFeature:
    """A curated sequence feature marking a sorting signal or membrane segment."""

    kind: str  # value from SIGNAL_FEATURE_TYPES
    start: int  # 1-based, inclusive (UniProt convention)
    end: int  # 1-based, inclusive
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "description": self.description,
        }


@dataclass
class ProteinRecord:
    """A single curated protein with localization labels and sorting signals."""

    accession: str
    entry_name: str
    sequence: str
    organism: str
    kingdom: str
    labels: list[str] = field(default_factory=list)
    features: list[SignalFeature] = field(default_factory=list)
    unmapped_terms: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> dict:
        return {
            "accession": self.accession,
            "entry_name": self.entry_name,
            "sequence": self.sequence,
            "organism": self.organism,
            "kingdom": self.kingdom,
            "length": self.length,
            "labels": sorted(self.labels),
            "features": [f.to_dict() for f in self.features],
            "unmapped_terms": sorted(self.unmapped_terms),
        }


def download_swissprot(dest_dir: Path | str, release: str = "current_release") -> Path:
    """Download ``uniprot_sprot.dat.gz`` unless it is already present.

    Returns the path to the local archive. The UniProt release string is recorded
    alongside it in ``swissprot_release.txt`` so the exact data version can be
    cited in the manuscript.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "uniprot_sprot.dat.gz"

    if target.exists() and target.stat().st_size > 0:
        logger.info("Swiss-Prot archive already present at %s (skipping download)", target)
        return target

    url = SWISSPROT_URL_TEMPLATE.format(release=release)
    logger.info("Downloading Swiss-Prot from %s", url)
    tmp = target.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with open(tmp, "wb") as handle, tqdm(
            total=total or None, unit="B", unit_scale=True, desc="uniprot_sprot.dat.gz"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                bar.update(len(chunk))
    shutil.move(tmp, target)

    _record_release(dest_dir, release)
    return target


def _record_release(dest_dir: Path, release: str) -> None:
    """Persist the UniProt release notes header for provenance."""
    try:
        response = requests.get(
            RELEASE_NOTE_URL_TEMPLATE.format(release=release), timeout=60
        )
        response.raise_for_status()
        header = "\n".join(response.text.splitlines()[:10])
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        logger.warning("Could not fetch release notes: %s", exc)
        header = f"release={release} (release notes unavailable)"
    (dest_dir / "swissprot_release.txt").write_text(header + "\n", encoding="utf-8")
    logger.info("Recorded Swiss-Prot provenance:\n%s", header)


def _parse_features(record) -> list[SignalFeature]:
    """Extract sorting-signal features from a Biopython ``SwissProt`` record.

    Biopython >= 1.79 exposes features as ``Bio.SeqFeature.SeqFeature`` objects
    with 0-based half-open ``location`` bounds; we convert back to the 1-based
    inclusive coordinates used in UniProt text and in our figures. Features with
    fuzzy or unknown positions are skipped, because residue-level alignment
    analyses require exact boundaries.
    """
    features: list[SignalFeature] = []
    for feature in record.features:
        kind = SIGNAL_FEATURE_TYPES.get(str(feature.type).upper())
        if kind is None:
            continue
        try:
            start = int(feature.location.start) + 1
            end = int(feature.location.end)
        except (TypeError, ValueError):
            continue  # fuzzy position (e.g. "<1" or "?")
        if start < 1 or end < start:
            continue
        description = ""
        if getattr(feature, "qualifiers", None):
            description = str(feature.qualifiers.get("note", "") or "")
        features.append(SignalFeature(kind=kind, start=start, end=end, description=description))
    return features


def parse_swissprot(
    dat_path: Path | str,
    require_experimental: bool = True,
    kingdoms: list[str] | None = None,
    limit: int | None = None,
) -> Iterator[ProteinRecord]:
    """Stream :class:`ProteinRecord` objects from ``uniprot_sprot.dat.gz``.

    Parameters
    ----------
    dat_path:
        Path to the (gzipped) Swiss-Prot flat file.
    require_experimental:
        Passed through to :func:`~psl_probe.data.labels.map_subcellular_location`.
    kingdoms:
        If given, only entries whose organism-classification lineage starts with
        one of these superkingdoms are yielded (e.g. ``["Eukaryota"]``).
    limit:
        Stop after this many *parsed* entries. Useful for smoke tests.

    Yields
    ------
    ProteinRecord
        One record per entry that carries at least one SUBCELLULAR LOCATION
        comment. Entries whose comment maps to no class are still yielded (with an
        empty ``labels`` list) so that mapping coverage can be audited; filtering
        happens in :func:`psl_probe.data.curate.curate_dataset`.
    """
    try:
        from Bio import SwissProt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Biopython is required to parse Swiss-Prot. Install it with "
            "`pip install biopython`."
        ) from exc

    dat_path = Path(dat_path)
    kingdom_filter = set(kingdoms) if kingdoms else None
    opener = gzip.open if dat_path.suffix == ".gz" else open

    n_yielded = 0
    with opener(dat_path, "rt", encoding="utf-8", errors="replace") as handle:
        for record in tqdm(SwissProt.parse(handle), desc="parsing Swiss-Prot", unit="entry"):
            lineage = record.organism_classification or []
            kingdom = lineage[0] if lineage else "Unknown"
            if kingdom_filter is not None and kingdom not in kingdom_filter:
                continue

            comments = [
                c.split("SUBCELLULAR LOCATION:", 1)[1]
                for c in record.comments
                if c.startswith("SUBCELLULAR LOCATION:")
            ]
            if not comments:
                continue

            classes: set[str] = set()
            unmapped: set[str] = set()
            for comment in comments:
                mapped, missed = map_subcellular_location(
                    comment, require_experimental=require_experimental
                )
                classes |= mapped
                unmapped |= missed

            yield ProteinRecord(
                accession=record.accessions[0],
                entry_name=record.entry_name,
                sequence=record.sequence.upper(),
                organism=record.organism.rstrip(". "),
                kingdom=kingdom,
                labels=sorted(classes),
                features=_parse_features(record),
                unmapped_terms=sorted(unmapped),
            )

            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                logger.info("Reached parse limit of %d entries", limit)
                return
