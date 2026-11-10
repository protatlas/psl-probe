"""Sequence ablations must have exactly the advertised effect on length and content."""

from __future__ import annotations

import pytest

from psl_probe.analysis.ablation import MASK_TOKEN, ablate_sequence

SEQ = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"


def test_nterm_delete_shortens_by_k() -> None:
    out = ablate_sequence(SEQ, "nterm_delete", 10)
    assert len(out) == len(SEQ) - 10
    assert out == SEQ[10:]


def test_nterm_mask_preserves_length() -> None:
    out = ablate_sequence(SEQ, "nterm_mask", 10)
    assert len(out) == len(SEQ)
    assert out[:10] == MASK_TOKEN * 10
    assert out[10:] == SEQ[10:]


def test_shuffle_preserves_length_and_composition() -> None:
    """The composition control must change order without changing residue counts."""
    out = ablate_sequence(SEQ, "shuffle_nterm", 20, seed=0)
    assert len(out) == len(SEQ)
    assert sorted(out) == sorted(SEQ)
    assert out[20:] == SEQ[20:]


def test_internal_delete_is_length_matched_to_nterm_delete() -> None:
    nterm = ablate_sequence(SEQ, "nterm_delete", 15)
    internal = ablate_sequence(SEQ, "internal_delete", 15)
    assert len(nterm) == len(internal)
    assert internal.startswith(SEQ[:5])  # N-terminus retained


def test_ablation_never_empties_a_short_sequence() -> None:
    short = "MKTII"
    out = ablate_sequence(short, "nterm_delete", 100)
    assert len(out) >= 1


def test_none_is_identity() -> None:
    assert ablate_sequence(SEQ, "none", 40) == SEQ
    assert ablate_sequence(SEQ, "nterm_delete", 0) == SEQ


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ablation mode"):
        ablate_sequence(SEQ, "reverse", 10)
