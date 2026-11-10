"""Attention-to-sorting-signal alignment.

Question addressed: do individual attention heads in a pLM specifically attend to
experimentally curated targeting signals (``SIGNAL`` / ``TRANSIT`` features), or is
apparent alignment just a consequence of these signals sitting at the N-terminus,
where attention is concentrated for position-related reasons?

We answer this with an *enrichment ratio*: the attention mass a head sends to
signal residues, divided by the mass expected if the head spread its attention
uniformly over the sequence. A ratio of 1 means no specificity. Crucially we also
compute the same statistic against a **position-matched null** -- a decoy window
of identical length placed at the same relative offset in proteins that carry no
curated signal -- so that a generic N-terminal attention bias cannot be mistaken
for signal recognition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionAlignment:
    """Per-head enrichment for one protein.

    ``enrichment`` has shape ``(n_layers, n_heads)``.
    """

    accession: str
    enrichment: np.ndarray
    signal_length: int
    sequence_length: int


def signal_mask_from_features(
    features: list[dict],
    sequence_length: int,
    kinds: tuple[str, ...] = ("signal_peptide", "transit_peptide"),
) -> np.ndarray | None:
    """Build a boolean residue mask over curated sorting-signal spans.

    UniProt coordinates are 1-based inclusive; the returned mask is 0-based.
    Returns ``None`` when no feature of the requested kinds is present.
    """
    mask = np.zeros(sequence_length, dtype=bool)
    found = False
    for feature in features:
        if feature["kind"] not in kinds:
            continue
        start = max(int(feature["start"]) - 1, 0)
        end = min(int(feature["end"]), sequence_length)
        if end > start:
            mask[start:end] = True
            found = True
    return mask if found else None


def attention_enrichment(
    attentions: tuple[Tensor, ...],
    residue_mask: Tensor,
    signal_masks: list[np.ndarray | None],
) -> list[np.ndarray | None]:
    """Compute per-head signal enrichment for a batch.

    Parameters
    ----------
    attentions:
        Tuple of ``(batch, heads, n_residues, n_residues)`` attention tensors, one
        per layer, already cropped to residue positions.
    residue_mask:
        ``(batch, n_residues)`` boolean mask of real residues.
    signal_masks:
        One entry per batch element: a boolean array over that protein's residues,
        or ``None`` to skip the protein.

    Returns
    -------
    list[numpy.ndarray | None]
        Per protein, an ``(n_layers, n_heads)`` enrichment array (or ``None``).

    Notes
    -----
    Attention rows are re-normalised over valid residue positions before the mass
    is summed, because cropping away special tokens removes a non-trivial share of
    each row's probability mass (the ``<cls>`` token in particular absorbs a large
    fraction of attention in ESM-2).
    """
    batch_size = residue_mask.shape[0]
    results: list[np.ndarray | None] = [None] * batch_size

    stacked = torch.stack(attentions, dim=0).float()  # (layers, batch, heads, R, R)
    valid = residue_mask.to(stacked.device)

    # Re-normalise each attention row over valid key positions.
    key_mask = valid[None, :, None, None, :]
    stacked = stacked * key_mask
    stacked = stacked / stacked.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    for b in range(batch_size):
        signal = signal_masks[b]
        if signal is None:
            continue
        length = int(valid[b].sum().item())
        if length == 0 or signal.sum() == 0 or signal.sum() >= length:
            continue

        signal_tensor = torch.zeros(valid.shape[1], dtype=torch.bool, device=stacked.device)
        usable = min(len(signal), length)
        signal_tensor[:usable] = torch.as_tensor(signal[:usable], device=stacked.device)

        # Mean over query positions (real residues only) of the mass sent to signal keys.
        mass = stacked[:, b, :, :, signal_tensor].sum(dim=-1)  # (layers, heads, R)
        query_weights = valid[b].float()
        observed = (mass * query_weights[None, None, :]).sum(dim=-1) / query_weights.sum()

        expected = float(signal_tensor.sum().item()) / length
        results[b] = (observed / expected).cpu().numpy()

    return results


def decoy_masks(
    signal_masks: list[np.ndarray | None],
    lengths: list[int],
    rng: np.random.Generator,
) -> list[np.ndarray | None]:
    """Position-matched decoy windows for the null model.

    For each protein that has a signal of length ``k`` starting at offset ``s``,
    the decoy is a window of the same length ``k`` placed uniformly at random in
    the region **downstream** of the signal. Matching length (but not position)
    isolates positional bias from sequence-content recognition.
    """
    decoys: list[np.ndarray | None] = []
    for signal, length in zip(signal_masks, lengths, strict=True):
        if signal is None or length <= 0:
            decoys.append(None)
            continue
        positions = np.flatnonzero(signal)
        if positions.size == 0:
            decoys.append(None)
            continue
        window = int(positions.size)
        downstream_start = int(positions.max()) + 1
        if downstream_start + window > length:
            decoys.append(None)
            continue
        start = int(rng.integers(downstream_start, length - window + 1))
        decoy = np.zeros(length, dtype=bool)
        decoy[start : start + window] = True
        decoys.append(decoy)
    return decoys


def bootstrap_enrichment(
    per_protein: list[np.ndarray],
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    family_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap the mean per-head enrichment over proteins.

    Unlike the probing analyses there is no fitted model here and therefore no
    probe-fitting variance: the encoder is frozen and deterministic. The two
    sources of uncertainty are which proteins were sampled and, for the decoy
    null, where each decoy window happened to land. This function addresses the
    first by resampling proteins; the caller averages over several decoy
    placements to address the second.

    Parameters
    ----------
    per_protein:
        One ``(n_layers, n_heads)`` enrichment array per protein.
    n_samples:
        Bootstrap resamples over proteins.
    confidence:
        Interval coverage.
    seed:
        Resampling seed.
    family_size:
        If given, additionally return Bonferroni-adjusted simultaneous intervals at
        level ``1 - (1 - confidence) / family_size``. Every layer-head pair is a
        separate test -- 660 of them for ESM-2 650M -- so an uncorrected 95%
        interval would be expected to exclude the null for roughly 33 heads by
        chance alone.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``mean``, ``lo``, ``hi``, each ``(n_layers, n_heads)``, plus ``n_proteins``;
        and ``lo_adj``/``hi_adj`` when ``family_size`` is given.
    """
    if not per_protein:
        raise ValueError("no per-protein enrichment arrays were supplied")

    stack = np.stack(per_protein, axis=0)            # (proteins, layers, heads)
    n_proteins = stack.shape[0]
    rng = np.random.default_rng(seed)

    draws = np.empty((n_samples, *stack.shape[1:]), dtype=np.float64)
    for b in range(n_samples):
        index = rng.integers(0, n_proteins, n_proteins)
        draws[b] = np.nanmean(stack[index], axis=0)

    alpha = (1.0 - confidence) / 2.0
    out = {
        "mean": np.nanmean(stack, axis=0),
        "lo": np.nanpercentile(draws, 100 * alpha, axis=0),
        "hi": np.nanpercentile(draws, 100 * (1 - alpha), axis=0),
        "n_proteins": np.array(n_proteins),
    }
    if family_size:
        alpha_adj = (1.0 - confidence) / family_size / 2.0
        out["lo_adj"] = np.nanpercentile(draws, 100 * alpha_adj, axis=0)
        out["hi_adj"] = np.nanpercentile(draws, 100 * (1 - alpha_adj), axis=0)
    return out


def summarise_enrichment(
    per_protein: list[np.ndarray], n_layers: int, n_heads: int
) -> dict[str, np.ndarray]:
    """Aggregate per-protein enrichment into layer/head summary statistics.

    Returns ``mean`` and ``median`` ``(n_layers, n_heads)`` arrays plus
    ``best_head_per_layer`` giving the maximum-enrichment head index per layer.
    """
    if not per_protein:
        empty = np.full((n_layers, n_heads), np.nan)
        return {"mean": empty, "median": empty.copy(),
                "best_head_per_layer": np.full(n_layers, -1, dtype=int)}

    stack = np.stack(per_protein, axis=0)  # (proteins, layers, heads)
    mean = np.nanmean(stack, axis=0)
    return {
        "mean": mean,
        "median": np.nanmedian(stack, axis=0),
        "n_proteins": np.array(len(per_protein)),
        "best_head_per_layer": np.nanargmax(mean, axis=1),
    }
