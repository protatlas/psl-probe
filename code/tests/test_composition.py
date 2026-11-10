"""Sequence-composition features used for the no-language-model baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psl_probe.analysis.composition import (
    FEATURE_SETS,
    KYTE_DOOLITTLE,
    PHYSCHEM_NAMES,
    RESIDUES,
    TM_WINDOW,
    amino_acid_composition,
    build_features,
    dipeptide_composition,
    physicochemical,
)


def test_composition_rows_are_normalised() -> None:
    out = amino_acid_composition(["ACDEF", "AAAA", "MKVLW"])
    assert out.shape == (3, 20)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_composition_counts_are_correct() -> None:
    out = amino_acid_composition(["AAAC"])
    assert out[0, RESIDUES.index("A")] == pytest.approx(0.75)
    assert out[0, RESIDUES.index("C")] == pytest.approx(0.25)
    assert out[0, RESIDUES.index("W")] == 0.0


def test_non_standard_residues_are_excluded_not_mapped() -> None:
    """X must not silently become a residue; it drops out of the denominator."""
    out = amino_acid_composition(["AAXC"])
    assert out[0, RESIDUES.index("A")] == pytest.approx(2 / 3)
    assert out[0, RESIDUES.index("C")] == pytest.approx(1 / 3)


def test_dipeptide_counts_adjacent_pairs_only() -> None:
    out = dipeptide_composition(["ACA"])
    size = len(RESIDUES)
    a, c = RESIDUES.index("A"), RESIDUES.index("C")
    assert out.shape == (1, 400)
    assert out[0, a * size + c] == pytest.approx(0.5)
    assert out[0, c * size + a] == pytest.approx(0.5)
    assert out[0, a * size + a] == 0.0


def test_dipeptide_skips_pairs_spanning_a_nonstandard_residue() -> None:
    """AXA contributes no dipeptide: neither AX, XA nor a fabricated AA."""
    out = dipeptide_composition(["AXA"])
    assert out.sum() == pytest.approx(0.0)


def test_gravy_matches_the_scale_mean() -> None:
    sequence = "AILVKR"
    out = physicochemical([sequence])
    expected = np.mean([KYTE_DOOLITTLE[r] for r in sequence])
    assert out[0, PHYSCHEM_NAMES.index("gravy")] == pytest.approx(expected, abs=1e-5)


def test_hydrophobic_window_finds_a_buried_helix() -> None:
    """A hydrophobic stretch inside a polar protein must raise the window score
    well above the whole-sequence mean, which is what makes the feature a
    transmembrane proxy rather than a restatement of GRAVY."""
    polar = "K" * 40
    helix = "L" * TM_WINDOW
    out = physicochemical([polar + helix + polar])
    gravy = out[0, PHYSCHEM_NAMES.index("gravy")]
    window = out[0, PHYSCHEM_NAMES.index("max_hydrophobic_window")]
    assert window == pytest.approx(KYTE_DOOLITTLE["L"], abs=1e-5)
    assert window > gravy + 3.0


def test_short_sequence_falls_back_to_the_mean() -> None:
    out = physicochemical(["LLL"])
    assert out[0, PHYSCHEM_NAMES.index("max_hydrophobic_window")] == pytest.approx(
        out[0, PHYSCHEM_NAMES.index("gravy")]
    )


def test_net_charge_has_the_expected_sign() -> None:
    positive = physicochemical(["KKKR"])[0, PHYSCHEM_NAMES.index("net_charge_per_residue")]
    negative = physicochemical(["DDDE"])[0, PHYSCHEM_NAMES.index("net_charge_per_residue")]
    assert positive == pytest.approx(1.0)
    assert negative == pytest.approx(-1.0)


@pytest.mark.parametrize("name,width", [("aac", 20), ("aac_physchem", 26),
                                        ("dipeptide", 400), ("all", 426)])
def test_feature_sets_have_the_declared_width(name: str, width: int) -> None:
    frame = pd.DataFrame({"sequence": ["ACDEFGHIKLMNPQRSTVWY" * 3, "MKVLWAAT" * 5]})
    matrix, names = build_features(frame, name)
    assert matrix.shape == (2, width)
    assert len(names) == width
    assert len(set(names)) == width


def test_unknown_feature_set_raises() -> None:
    frame = pd.DataFrame({"sequence": ["ACDEF"]})
    with pytest.raises(KeyError, match="unknown feature set"):
        build_features(frame, "not_a_set")


def test_every_declared_set_is_buildable() -> None:
    frame = pd.DataFrame({"sequence": ["ACDEFGHIKLMNPQRSTVWY" * 2]})
    for name in FEATURE_SETS:
        matrix, names = build_features(frame, name)
        assert matrix.shape[1] == len(names)
        assert np.isfinite(matrix).all()
