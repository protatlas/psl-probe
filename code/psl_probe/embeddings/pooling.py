"""Reduction of per-residue representations to fixed-size protein vectors.

The choice of pooling window is itself an experimental variable in this study.
``mean`` pooling gives the conventional whole-protein representation; ``nterm``
restricts the pool to the first ``w`` residues, the region that carries signal
peptides and mitochondrial/chloroplast transit peptides. Comparing probe
performance across these windows isolates *where along the sequence* a pLM stores
localization-predictive information.

.. note::
   All reductions are accumulated in ``float32`` even when the model runs in
   half precision. ESM-2 activations reach magnitudes of order :math:`10^2`, so
   summing over a 1000-residue protein reaches :math:`10^5` and overflows the
   fp16 range (65504) to ``inf`` -- even though the resulting *mean* is small
   and perfectly representable. Accumulating in fp16 silently corrupts exactly
   the long proteins in the deepest layers, which is where the interesting
   signal lives.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Pooling strategies understood by :func:`pool_representations`.
STRATEGIES = ("mean", "cls", "max", "nterm", "cterm")


#: Reductions are accumulated in this dtype regardless of the model's precision.
_ACCUM_DTYPE = torch.float32


def _accumulate(hidden: Tensor) -> Tensor:
    """Upcast half-precision activations so reductions cannot overflow."""
    if hidden.dtype in (torch.float16, torch.bfloat16):
        return hidden.to(_ACCUM_DTYPE)
    return hidden


def _masked_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    """Mean over the ``True`` positions of ``mask``.

    Parameters
    ----------
    hidden:
        ``(n_layers, batch, n_residues, hidden)``.
    mask:
        ``(batch, n_residues)`` boolean.

    Returns
    -------
    Tensor
        ``(n_layers, batch, hidden)`` in ``float32``. Sequences with an
        all-``False`` mask (which can occur when a window falls entirely beyond a
        short protein) yield zeros rather than NaNs.
    """
    hidden = _accumulate(hidden)
    weights = mask[None, :, :, None].to(hidden.dtype)
    total = (hidden * weights).sum(dim=2)
    counts = weights.sum(dim=2).clamp(min=1.0)
    return total / counts


def _window_mask(mask: Tensor, window: int, from_end: bool) -> Tensor:
    """Restrict ``mask`` to the first (or last) ``window`` real residues."""
    lengths = mask.sum(dim=1)
    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
    if from_end:
        start = (lengths[:, None] - window).clamp(min=0)
        return mask & (positions >= start)
    return mask & (positions < window)


def pool_representations(
    hidden: Tensor,
    mask: Tensor,
    strategies: list[str],
    nterm_window: int = 70,
    cterm_window: int = 30,
) -> dict[str, Tensor]:
    """Pool residue-level hidden states into per-protein vectors.

    Parameters
    ----------
    hidden:
        ``(n_layers, batch, n_residues, hidden)`` residue-aligned hidden states.
    mask:
        ``(batch, n_residues)`` boolean residue mask.
    strategies:
        Subset of :data:`STRATEGIES`.
    nterm_window, cterm_window:
        Window lengths in residues for the terminal pools.

    Returns
    -------
    dict[str, Tensor]
        Strategy name to ``(n_layers, batch, hidden)`` tensor in ``float32``
        (see the module note on accumulation precision).
    """
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown pooling strategies: {sorted(unknown)}")

    pooled: dict[str, Tensor] = {}
    for strategy in strategies:
        if strategy == "mean":
            pooled[strategy] = _masked_mean(hidden, mask)
        elif strategy == "cls":
            # Position 0 of the residue-aligned tensor, i.e. the first residue.
            # (The model's own <cls> token is stripped during alignment; using the
            # first residue keeps the definition uniform across model families.)
            pooled[strategy] = _accumulate(hidden[:, :, 0, :]).clone()
        elif strategy == "max":
            promoted = _accumulate(hidden)
            neg_inf = torch.finfo(promoted.dtype).min
            masked = promoted.masked_fill(~mask[None, :, :, None], neg_inf)
            pooled[strategy] = masked.amax(dim=2)
        elif strategy == "nterm":
            pooled[strategy] = _masked_mean(hidden, _window_mask(mask, nterm_window, False))
        elif strategy == "cterm":
            pooled[strategy] = _masked_mean(hidden, _window_mask(mask, cterm_window, True))
    return pooled
