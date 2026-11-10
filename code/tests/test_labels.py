"""Tests for the Swiss-Prot annotation mapping.

Every case below is a real Swiss-Prot ``SUBCELLULAR LOCATION`` comment pattern.
These tests are the audit trail for the label taxonomy: if a mapping rule changes,
the effect on the benchmark is visible here.
"""

from __future__ import annotations

import numpy as np
import pytest

from psl_probe.data.labels import (
    LOCALIZATION_CLASSES,
    labels_to_vector,
    map_subcellular_location,
    vector_to_labels,
)


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("Mitochondrion matrix {ECO:0000269|PubMed:1}.", {"Mitochondrion"}),
        ("Secreted {ECO:0000269|PubMed:1}.", {"Extracellular"}),
        ("Nucleus {ECO:0000269|PubMed:1}. Cytoplasm {ECO:0000269|PubMed:2}.",
         {"Nucleus", "Cytoplasm"}),
        # Nested compartments must resolve to the innermost taxonomy class.
        ("Plastid, chloroplast thylakoid membrane {ECO:0000269|PubMed:1}.", {"Plastid"}),
        ("Golgi apparatus, trans-Golgi network membrane {ECO:0000269|PubMed:1}.",
         {"Golgi apparatus"}),
        # Topology qualifiers describe insertion, not compartment identity.
        ("Cell membrane {ECO:0000269|PubMed:1}; Single-pass type I membrane protein.",
         {"Cell membrane"}),
        ("Peroxisome matrix {ECO:0000269|PubMed:1}.", {"Peroxisome"}),
        ("Lysosome lumen {ECO:0000269|PubMed:1}.", {"Lysosome/Vacuole"}),
        ("Endoplasmic reticulum lumen {ECO:0000269|PubMed:1}.", {"Endoplasmic reticulum"}),
    ],
)
def test_maps_known_terms(comment: str, expected: set[str]) -> None:
    classes, _ = map_subcellular_location(comment)
    assert classes == expected


def test_rejects_non_experimental_evidence() -> None:
    """ECO:0000250 is sequence-similarity inference and must not produce a label."""
    classes, _ = map_subcellular_location("Cytoplasm {ECO:0000250}.")
    assert classes == set()
    classes, _ = map_subcellular_location(
        "Cytoplasm {ECO:0000250}.", require_experimental=False
    )
    assert classes == {"Cytoplasm"}


def test_ambiguous_terms_are_reported_not_guessed() -> None:
    """A bare 'Membrane' names no compartment and must be surfaced, not assigned."""
    classes, unmapped = map_subcellular_location("Membrane {ECO:0000269|PubMed:1}.")
    assert classes == set()
    assert unmapped == {"Membrane"}


def test_note_blocks_are_ignored() -> None:
    """Free prose in Note= must not contribute labels."""
    comment = (
        "Cytoplasm {ECO:0000269|PubMed:1}. Note=Translocates to the nucleus upon "
        "stimulation with mitochondrion-derived signals."
    )
    classes, _ = map_subcellular_location(comment)
    assert classes == {"Cytoplasm"}


def test_label_vector_roundtrip() -> None:
    labels = ["Nucleus", "Peroxisome"]
    vector = labels_to_vector(labels)
    assert vector.shape == (len(LOCALIZATION_CLASSES),)
    assert vector.sum() == 2
    assert sorted(vector_to_labels(vector)) == sorted(labels)


def test_unknown_class_raises() -> None:
    with pytest.raises(KeyError):
        labels_to_vector(["Ribosome"])


def test_class_order_is_stable() -> None:
    """Figures and stored label matrices depend on this exact ordering."""
    assert LOCALIZATION_CLASSES[0] == "Cytoplasm"
    assert LOCALIZATION_CLASSES[-1] == "Peroxisome"
    assert len(LOCALIZATION_CLASSES) == len(set(LOCALIZATION_CLASSES)) == 10
    assert np.all(labels_to_vector([]) == 0)
