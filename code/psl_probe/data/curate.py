"""Turn parsed Swiss-Prot entries into the analysis-ready benchmark table."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from psl_probe.config import Config
from psl_probe.data.labels import LOCALIZATION_CLASSES
from psl_probe.data.uniprot import AMBIGUOUS_RESIDUES, VALID_RESIDUES, ProteinRecord
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

#: Reasons a record can be discarded, reported as a filter cascade in Table 1.
FILTER_REASONS = (
    "no_mapped_label",
    "too_many_labels",
    "length_out_of_range",
    "non_standard_residues",
)


def curate_dataset(
    records: Iterable[ProteinRecord],
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply inclusion criteria and assemble the benchmark table.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        ``(dataset, filter_report, unmapped)``.

        ``dataset`` has one row per retained protein with columns
        ``accession, entry_name, sequence, length, organism, kingdom``, one binary
        column per class in :data:`LOCALIZATION_CLASSES`, plus ``features`` (list of
        dicts) and ``n_labels``.

        ``filter_report`` counts how many records each criterion removed, in
        cascade order.

        ``unmapped`` lists location terms that matched no taxonomy rule, with their
        frequency. This is the audit trail for mapping coverage: a term appearing
        near the top is a candidate for a new rule in
        :mod:`psl_probe.data.labels`, and reporting it is what keeps the mapping
        honest rather than silently lossy.
    """
    kept: list[dict] = []
    removed = Counter()
    unmapped_terms = Counter()

    for record in records:
        unmapped_terms.update(record.unmapped_terms)

        if not record.labels:
            removed["no_mapped_label"] += 1
            continue
        if len(record.labels) > cfg.data.max_labels_per_protein:
            removed["too_many_labels"] += 1
            continue
        if not (cfg.data.min_length <= record.length <= cfg.data.max_length):
            removed["length_out_of_range"] += 1
            continue

        residues = set(record.sequence)
        if not residues <= (VALID_RESIDUES | AMBIGUOUS_RESIDUES):
            removed["non_standard_residues"] += 1
            continue

        row: dict = {
            "accession": record.accession,
            "entry_name": record.entry_name,
            "sequence": record.sequence,
            "length": record.length,
            "organism": record.organism,
            "kingdom": record.kingdom,
            "features": [f.to_dict() for f in record.features],
            "n_labels": len(record.labels),
        }
        for cls in LOCALIZATION_CLASSES:
            row[cls] = int(cls in record.labels)
        kept.append(row)

    dataset = pd.DataFrame(kept)
    filter_report = pd.DataFrame(
        {"reason": FILTER_REASONS, "n_removed": [removed[r] for r in FILTER_REASONS]}
    )

    logger.info("Retained %d proteins; removed %d", len(dataset), sum(removed.values()))
    for reason in FILTER_REASONS:
        logger.info("  filtered %-24s %d", reason, removed[reason])

    unmapped = pd.DataFrame(
        unmapped_terms.most_common(), columns=["term", "n_occurrences"]
    )
    if unmapped_terms:
        top = ", ".join(f"{t} ({n})" for t, n in unmapped_terms.most_common(10))
        logger.info("Top unmapped location terms: %s", top)

    return dataset, filter_report, unmapped


def summarise_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and multi-label statistics for the manuscript's Table 1."""
    rows = []
    n_total = len(dataset)
    for cls in LOCALIZATION_CLASSES:
        n_positive = int(dataset[cls].sum())
        # Proteins that carry this class *and* at least one other.
        multi = int(((dataset[cls] == 1) & (dataset["n_labels"] > 1)).sum())
        rows.append(
            {
                "class": cls,
                "n_proteins": n_positive,
                "prevalence": n_positive / n_total if n_total else 0.0,
                "n_multi_label": multi,
                "frac_multi_label": multi / n_positive if n_positive else 0.0,
            }
        )
    return pd.DataFrame(rows)


def signal_feature_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Count curated sorting-signal features, broken down by localization class.

    This table motivates the attention-alignment analysis: it quantifies how many
    proteins per compartment carry an experimentally curated ``SIGNAL`` or
    ``TRANSIT`` annotation that can serve as residue-level ground truth.
    """
    rows = []
    for cls in LOCALIZATION_CLASSES:
        subset = dataset[dataset[cls] == 1]
        counts = Counter()
        for feats in subset["features"]:
            for kind in {f["kind"] for f in feats}:
                counts[kind] += 1
        row = {"class": cls, "n_proteins": len(subset)}
        row.update({kind: counts.get(kind, 0) for kind in
                    ("signal_peptide", "transit_peptide", "transmembrane",
                     "propeptide", "lipid_anchor")})
        rows.append(row)
    return pd.DataFrame(rows)


def save_dataset(dataset: pd.DataFrame, path: Path | str) -> Path:
    """Persist the benchmark as Parquet (preserves the nested ``features`` column)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)
    logger.info("Wrote %d proteins to %s", len(dataset), path)
    return path


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Read a benchmark table written by :func:`save_dataset`."""
    return pd.read_parquet(path)
