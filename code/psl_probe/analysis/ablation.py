"""Sequence-ablation experiments.

If a pLM's localization signal really comes from N-terminal targeting peptides,
deleting the N-terminus should degrade probe performance *selectively*: sharply for
compartments reached by an N-terminal signal (mitochondrion, plastid, ER,
extracellular) and much less for compartments whose determinants are internal or
C-terminal (nucleus via NLS, peroxisome via the C-terminal PTS1 tripeptide).
A uniform degradation across all compartments would instead indicate that the
ablation merely damages the representation in general.

We therefore always report the *per-compartment* delta and contrast it against a
length-matched internal deletion, which removes the same number of residues from
the middle of the protein and thus controls for the loss of sequence length alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

#: Ablation modes understood by :func:`ablate_sequence`.
MODES = ("none", "nterm_delete", "nterm_mask", "internal_delete", "cterm_delete", "shuffle_nterm")

#: Token substituted by masking ablations; ``X`` is the standard UniProt code for
#: an unspecified residue and is in-vocabulary for both ESM-2 and ProtT5.
MASK_TOKEN = "X"


def ablate_sequence(sequence: str, mode: str, n_residues: int, seed: int = 0) -> str:
    """Return an ablated copy of ``sequence``.

    Parameters
    ----------
    sequence:
        Amino-acid string.
    mode:
        One of :data:`MODES`. ``"none"`` returns the sequence unchanged.
    n_residues:
        Number of residues affected.
    seed:
        Seed for ``shuffle_nterm``.

    Notes
    -----
    ``nterm_delete`` shortens the protein, whereas ``nterm_mask`` preserves length
    by substituting ``X``. Both are reported: deletion is the biologically faithful
    analogue of removing a cleaved presequence, while masking holds sequence length
    -- a variable pLMs are known to encode -- constant.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown ablation mode {mode!r}; expected one of {MODES}")
    if mode == "none" or n_residues <= 0:
        return sequence

    length = len(sequence)
    k = min(n_residues, max(length - 1, 0))
    if k <= 0:
        return sequence

    if mode == "nterm_delete":
        return sequence[k:]
    if mode == "nterm_mask":
        return MASK_TOKEN * k + sequence[k:]
    if mode == "cterm_delete":
        return sequence[: length - k]
    if mode == "internal_delete":
        start = max((length - k) // 2, 0)
        return sequence[:start] + sequence[start + k :]
    if mode == "shuffle_nterm":
        rng = np.random.default_rng(seed)
        prefix = list(sequence[:k])
        rng.shuffle(prefix)
        return "".join(prefix) + sequence[k:]
    raise AssertionError("unreachable")


def build_ablated_dataset(
    dataset: pd.DataFrame, mode: str, n_residues: int, seed: int = 0
) -> pd.DataFrame:
    """Copy the benchmark with ``sequence`` replaced by its ablated form."""
    out = dataset.copy()
    out["sequence"] = [
        ablate_sequence(s, mode, n_residues, seed=seed + i)
        for i, s in enumerate(dataset["sequence"])
    ]
    out["length"] = out["sequence"].str.len()
    logger.info(
        "Ablation %s(n=%d): mean length %.1f -> %.1f",
        mode, n_residues, dataset["length"].mean(), out["length"].mean(),
    )
    return out


def ablation_deltas(
    baseline: Sequence[dict], ablated: Sequence[dict], metric: str = "mcc"
) -> pd.DataFrame:
    """Per-compartment change in a metric between baseline and ablated probes.

    Parameters
    ----------
    baseline, ablated:
        Per-class metric records as returned by
        :func:`~psl_probe.analysis.metrics.per_class_metrics`.
    metric:
        Key within those records to compare.

    Returns
    -------
    pandas.DataFrame
        Columns ``class``, ``baseline``, ``ablated``, ``delta``, ``relative_delta``.
    """
    base_map = {r["class"]: r[metric] for r in baseline}
    abl_map = {r["class"]: r[metric] for r in ablated}
    rows = []
    for name, base_value in base_map.items():
        ablated_value = abl_map.get(name, np.nan)
        delta = ablated_value - base_value
        rows.append(
            {
                "class": name,
                "baseline": base_value,
                "ablated": ablated_value,
                "delta": delta,
                "relative_delta": delta / base_value if base_value not in (0, np.nan) else np.nan,
            }
        )
    return pd.DataFrame(rows)
