"""Explicit sequence-composition features, as a baseline for the probing results.

The manuscript attributes the decodability of the compartments that resist
N-terminal explanation to "distributed, compositional" determinants, and names
hydrophobicity, length and cysteine content specifically. This module computes
those properties directly, so that claim can be measured rather than asserted:
a probe fitted to these features alone bounds how much of the decodable signal
is recoverable from composition without any learned representation.

Feature sets are built from the amino-acid sequence only. Nothing here looks at
a language model, so the resulting scores are a floor that any representation
claiming to encode more than composition must clear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

#: The twenty standard residues, in a fixed order so feature columns are stable
#: across runs. Non-standard codes (X, B, Z, U, O) are ignored rather than
#: mapped, which leaves them out of the composition denominator; the benchmark
#: filters entries containing them, so this affects nothing in practice.
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

#: Kyte & Doolittle (1982) hydropathy index, J. Mol. Biol. 157:105-132. Used for
#: the GRAVY score and for the sliding-window transmembrane-helix proxy.
KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

#: Window length for the hydrophobic-stretch statistic. Nineteen residues is the
#: canonical span of a membrane-crossing alpha helix and is what hydropathy plots
#: have used since Kyte & Doolittle; it is a crude transmembrane detector, which
#: is the point -- it tests whether a crude detector suffices.
TM_WINDOW = 19

PHYSCHEM_NAMES = (
    "gravy",
    "max_hydrophobic_window",
    "frac_charged",
    "net_charge_per_residue",
    "aromaticity",
    "log10_length",
)

#: Named feature sets. Each maps to the blocks it concatenates, so the
#: incremental value of each block is readable off the results table.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "aac": ("aac",),
    "aac_physchem": ("aac", "physchem"),
    "dipeptide": ("dipeptide",),
    "all": ("aac", "dipeptide", "physchem"),
}


def amino_acid_composition(sequences: list[str]) -> np.ndarray:
    """Return ``(n, 20)`` residue frequencies, each row summing to at most one."""
    index = {residue: i for i, residue in enumerate(RESIDUES)}
    out = np.zeros((len(sequences), len(RESIDUES)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for residue in sequence:
            position = index.get(residue)
            if position is not None:
                out[row, position] += 1.0
        total = out[row].sum()
        if total > 0:
            out[row] /= total
    return out


def dipeptide_composition(sequences: list[str]) -> np.ndarray:
    """Return ``(n, 400)`` adjacent-pair frequencies in fixed residue order.

    Dipeptides capture local order that single-residue composition discards, so
    the contrast between this and :func:`amino_acid_composition` separates
    "composition" from "very short-range order".
    """
    index = {residue: i for i, residue in enumerate(RESIDUES)}
    size = len(RESIDUES)
    out = np.zeros((len(sequences), size * size), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        first = index.get(sequence[0]) if sequence else None
        for residue in sequence[1:]:
            second = index.get(residue)
            if first is not None and second is not None:
                out[row, first * size + second] += 1.0
            first = second
        total = out[row].sum()
        if total > 0:
            out[row] /= total
    return out


def physicochemical(sequences: list[str]) -> np.ndarray:
    """Return ``(n, 6)`` aggregate properties the manuscript appeals to by name.

    These are the quantities the Discussion invokes when explaining the
    compartments it cannot localise to the N-terminus: hydrophobicity and length
    for membrane proteins, and global composition for cytosolic ones.
    """
    out = np.zeros((len(sequences), len(PHYSCHEM_NAMES)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        hydropathy = np.array(
            [KYTE_DOOLITTLE.get(residue, 0.0) for residue in sequence], dtype=np.float32
        )
        length = max(len(sequence), 1)
        out[row, 0] = hydropathy.mean() if hydropathy.size else 0.0
        if hydropathy.size >= TM_WINDOW:
            # Sliding-window mean via a cumulative sum: O(L) rather than O(L*W).
            cumulative = np.concatenate([[0.0], np.cumsum(hydropathy)])
            windows = (cumulative[TM_WINDOW:] - cumulative[:-TM_WINDOW]) / TM_WINDOW
            out[row, 1] = windows.max()
        else:
            out[row, 1] = out[row, 0]
        counts = {residue: sequence.count(residue) for residue in "DEKRFWY"}
        charged = counts["D"] + counts["E"] + counts["K"] + counts["R"]
        out[row, 2] = charged / length
        out[row, 3] = (counts["K"] + counts["R"] - counts["D"] - counts["E"]) / length
        out[row, 4] = (counts["F"] + counts["W"] + counts["Y"]) / length
        out[row, 5] = np.log10(length)
    return out


def build_features(dataset: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Assemble one named feature set, returning the matrix and its column names.

    Parameters
    ----------
    dataset:
        Benchmark table carrying a ``sequence`` column, in the row order the
        caller will use for labels and split indices.
    feature_set:
        A key of :data:`FEATURE_SETS`.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {feature_set!r}; expected one of {list(FEATURE_SETS)}")
    sequences = dataset["sequence"].tolist()
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for block in FEATURE_SETS[feature_set]:
        if block == "aac":
            blocks.append(amino_acid_composition(sequences))
            names += [f"aac_{r}" for r in RESIDUES]
        elif block == "dipeptide":
            blocks.append(dipeptide_composition(sequences))
            names += [f"dip_{a}{b}" for a in RESIDUES for b in RESIDUES]
        elif block == "physchem":
            blocks.append(physicochemical(sequences))
            names += list(PHYSCHEM_NAMES)
        else:  # pragma: no cover - FEATURE_SETS is the only source of block names
            raise AssertionError(f"unhandled feature block {block!r}")
    matrix = np.concatenate(blocks, axis=1)
    logger.info("Feature set %s: %d features over %d proteins",
                feature_set, matrix.shape[1], matrix.shape[0])
    return matrix, names
