"""Label taxonomy and UniProt annotation mapping.

We adopt the ten-compartment multi-label taxonomy established by DeepLoc 2.0
(Thumuluri et al., NAR 2022) so that our probing results are directly comparable
to the supervised-predictor literature. Swiss-Prot ``CC -!- SUBCELLULAR LOCATION``
free text is mapped onto this taxonomy with the ordered keyword rules below.

The mapping is deliberately conservative: a term that cannot be assigned
unambiguously (e.g. a bare "Membrane" or "Cytoplasmic vesicle") yields no label
rather than a guessed one, and such proteins are dropped downstream. Every rule
here is auditable and unit-tested in ``tests/test_labels.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

#: Canonical class order. All label vectors and figures use this ordering.
LOCALIZATION_CLASSES: tuple[str, ...] = (
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
)

CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(LOCALIZATION_CLASSES)}

#: Short labels for compact figure axes.
CLASS_ABBREVIATIONS: dict[str, str] = {
    "Cytoplasm": "Cyto",
    "Nucleus": "Nucl",
    "Extracellular": "Extr",
    "Cell membrane": "CeMe",
    "Mitochondrion": "Mito",
    "Plastid": "Plas",
    "Endoplasmic reticulum": "ER",
    "Lysosome/Vacuole": "LyVa",
    "Golgi apparatus": "Golg",
    "Peroxisome": "Pero",
}

#: UniProt sequence-feature keys that mark an experimentally curated sorting signal.
#: These drive the attention-alignment and N-terminal ablation analyses.
SIGNAL_FEATURE_TYPES: dict[str, str] = {
    "SIGNAL": "signal_peptide",
    "TRANSIT": "transit_peptide",  # mitochondrial / chloroplast / thylakoid
    "TRANSMEM": "transmembrane",
    "PROPEP": "propeptide",
    "LIPID": "lipid_anchor",  # includes GPI-anchor amidated residues
}

#: ECO codes accepted as experimental evidence for a localization annotation.
#: ECO:0000269 = experimental evidence used in manual assertion;
#: ECO:0007744 = combinatorial evidence used in manual assertion (e.g. HTP + curation).
EXPERIMENTAL_ECO_CODES: frozenset[str] = frozenset({"ECO:0000269", "ECO:0007744"})

# ---------------------------------------------------------------------------
# Ordered mapping rules.
#
# Each entry is (compiled pattern, class name). Rules are evaluated in order and
# the FIRST match wins, so more specific compartments must precede the generic
# ones they contain (e.g. "nucleus membrane" before "membrane" heuristics, and
# "chloroplast thylakoid" before "chloroplast").
# ---------------------------------------------------------------------------
_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), cls)
    for pattern, cls in (
        # -- Plastid (must precede generic membrane/lumen rules) ------------
        (r"\b(plastid|chloroplast|chromoplast|amyloplast|apicoplast|thylakoid)\b", "Plastid"),
        # -- Mitochondrion --------------------------------------------------
        (r"\bmitochondri(on|al|a)\b", "Mitochondrion"),
        (r"\bhydrogenosome\b", "Mitochondrion"),
        # -- Peroxisome -----------------------------------------------------
        (r"\b(peroxisom(e|al)|glyoxysom(e|al)|glycosom(e|al))\b", "Peroxisome"),
        # -- Lysosome / Vacuole ---------------------------------------------
        (r"\b(lysosom(e|al)|vacuol(e|ar)|melanosom(e|al)|autophagosom(e|al))\b", "Lysosome/Vacuole"),
        # -- Golgi ----------------------------------------------------------
        (r"\b(golgi|trans-golgi|cis-golgi)\b", "Golgi apparatus"),
        # -- Endoplasmic reticulum ------------------------------------------
        (r"\b(endoplasmic reticulum|sarcoplasmic reticulum|microsom(e|al))\b",
         "Endoplasmic reticulum"),
        # -- Nucleus (nucleolus, nucleoplasm, chromosome are nuclear) --------
        (r"\b(nucleus|nuclear|nucleolus|nucleoplasm|chromosome|nucleoid)\b", "Nucleus"),
        # -- Extracellular ---------------------------------------------------
        (r"\b(secreted|extracellular|cell wall|periplasm|apoplast)\b", "Extracellular"),
        # -- Cell membrane ---------------------------------------------------
        (r"\b(cell membrane|plasma membrane|cell surface|cell projection|"
         r"cell junction|apical cell membrane|basolateral cell membrane)\b", "Cell membrane"),
        # -- Cytoplasm (cytosol, cytoskeleton) -------------------------------
        (r"\b(cytoplasm|cytosol|cytoskeleton|myofibril|sarcomere|"
         r"perinuclear region|stress granule|p-body)\b", "Cytoplasm"),
    )
)

#: Terms that are real UniProt locations but deliberately map to nothing, because
#: they are either compartment-ambiguous or outside the ten-class taxonomy.
_UNMAPPED_TERMS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*membrane\s*$",
        r"^\s*cytoplasmic vesicle\b",
        r"^\s*endosom(e|al)\b",
        r"^\s*early endosome\b",
        r"^\s*late endosome\b",
        r"^\s*recycling endosome\b",
        r"^\s*virion\b",
        r"^\s*host\b",
        r"^\s*cilium\b",
        r"^\s*flagellum\b",
    )
)

#: Qualifiers stripped before matching (topology, not compartment identity).
_QUALIFIER_PATTERN = re.compile(
    r"\b(single-pass|multi-pass|type i|type ii|type iii|type iv|"
    r"lipid-anchor|gpi-anchor|peripheral|integral|gpi-like-anchor|"
    r"cytoplasmic side|lumenal side|extracellular side|anchor)\b",
    re.IGNORECASE,
)

#: Evidence tag such as "{ECO:0000269|PubMed:12345678}".
_EVIDENCE_PATTERN = re.compile(r"\{([^}]*)\}")

#: Note blocks are free prose and are not parsed for locations.
_NOTE_PATTERN = re.compile(r"\bNote=.*$", re.IGNORECASE | re.DOTALL)


def _strip_evidence(text: str) -> tuple[str, set[str]]:
    """Split a location string into its bare term and the set of ECO codes."""
    codes: set[str] = set()
    for match in _EVIDENCE_PATTERN.finditer(text):
        codes.update(re.findall(r"ECO:\d{7}", match.group(1)))
    bare = _EVIDENCE_PATTERN.sub("", text)
    return bare.strip(" .;,"), codes


def map_subcellular_location(
    comment: str,
    require_experimental: bool = True,
    experimental_codes: Iterable[str] = EXPERIMENTAL_ECO_CODES,
) -> tuple[set[str], set[str]]:
    """Map a Swiss-Prot ``SUBCELLULAR LOCATION`` comment to taxonomy classes.

    Parameters
    ----------
    comment:
        The comment body, i.e. everything after ``SUBCELLULAR LOCATION: ``.
        Multiple locations are separated by ``.`` or ``;`` in Swiss-Prot.
    require_experimental:
        If ``True``, a location only contributes a label when at least one of its
        evidence tags is in ``experimental_codes``. Locations with no evidence tag
        are rejected under this setting.
    experimental_codes:
        ECO codes accepted as experimental.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(classes, unmapped_terms)`` where ``classes`` are taxonomy names and
        ``unmapped_terms`` are location strings that matched no rule. The latter
        is returned so the curation script can report mapping coverage instead of
        silently discarding annotations.
    """
    accepted = set(experimental_codes)
    classes: set[str] = set()
    unmapped: set[str] = set()

    body = _NOTE_PATTERN.sub("", comment)

    for raw_term in re.split(r"[.;]", body):
        term, codes = _strip_evidence(raw_term)
        if not term:
            continue
        if require_experimental and not (codes & accepted):
            continue

        cleaned = _QUALIFIER_PATTERN.sub(" ", term)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
        if not cleaned:
            continue

        if any(pattern.search(cleaned) for pattern in _UNMAPPED_TERMS):
            unmapped.add(cleaned)
            continue

        for pattern, cls in _RULES:
            if pattern.search(cleaned):
                classes.add(cls)
                break
        else:
            unmapped.add(cleaned)

    return classes, unmapped


def labels_to_vector(labels: Sequence[str]) -> np.ndarray:
    """Convert class names to a multi-hot vector over :data:`LOCALIZATION_CLASSES`."""
    vector = np.zeros(len(LOCALIZATION_CLASSES), dtype=np.float32)
    for label in labels:
        try:
            vector[CLASS_TO_INDEX[label]] = 1.0
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown localization class {label!r}") from exc
    return vector


def vector_to_labels(vector: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Inverse of :func:`labels_to_vector` for a single sample."""
    return [name for i, name in enumerate(LOCALIZATION_CLASSES) if vector[i] >= threshold]
