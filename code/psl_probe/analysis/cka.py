"""Centred Kernel Alignment between layer representations.

CKA (Kornblith et al., ICML 2019) quantifies how similar two representations of
the same inputs are, invariant to orthogonal transformation and isotropic scaling.
We use the linear (unbiased HSIC) estimator, which is unbiased with respect to
sample size and therefore comparable across the differently sized subsamples used
for different model scales.

In this study CKA answers a question the probe accuracy alone cannot: whether the
layer at which localization becomes linearly decodable coincides with a genuine
change in representational geometry, or merely with a re-scaling of information
already present in earlier layers.
"""

from __future__ import annotations

import numpy as np

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def _center_gram(gram: np.ndarray) -> np.ndarray:
    """Double-centre a Gram matrix."""
    n = gram.shape[0]
    unit = np.ones((n, n), dtype=gram.dtype) / n
    return gram - unit @ gram - gram @ unit + unit @ gram @ unit


def unbiased_hsic(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al., 2012) for two Gram matrices."""
    n = gram_x.shape[0]
    if n < 4:
        raise ValueError("Unbiased HSIC requires at least 4 samples.")
    kx = gram_x.copy()
    ky = gram_y.copy()
    np.fill_diagonal(kx, 0.0)
    np.fill_diagonal(ky, 0.0)

    term_1 = float(np.sum(kx * ky))
    term_2 = float(kx.sum() * ky.sum()) / ((n - 1) * (n - 2))
    term_3 = 2.0 * float(kx.sum(axis=0) @ ky.sum(axis=0)) / (n - 2)
    return (term_1 + term_2 - term_3) / (n * (n - 3))


def linear_cka(features_x: np.ndarray, features_y: np.ndarray, unbiased: bool = True) -> float:
    """Linear CKA between two ``(n_samples, dim)`` representations of the same inputs.

    Returns a value in ``[0, 1]``; 1 means the representations are identical up to
    rotation and scaling.
    """
    if features_x.shape[0] != features_y.shape[0]:
        raise ValueError("CKA requires the same samples in both representations.")

    x = np.asarray(features_x, dtype=np.float64)
    y = np.asarray(features_y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    gram_x, gram_y = x @ x.T, y @ y.T
    if unbiased:
        hsic_xy = unbiased_hsic(gram_x, gram_y)
        hsic_xx = unbiased_hsic(gram_x, gram_x)
        hsic_yy = unbiased_hsic(gram_y, gram_y)
    else:
        gram_x, gram_y = _center_gram(gram_x), _center_gram(gram_y)
        hsic_xy = float(np.sum(gram_x * gram_y))
        hsic_xx = float(np.sum(gram_x * gram_x))
        hsic_yy = float(np.sum(gram_y * gram_y))

    denominator = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denominator <= 0:
        return float("nan")
    return float(hsic_xy / denominator)


def layer_cka_matrix(
    embeddings: np.ndarray,
    max_samples: int = 4000,
    seed: int = 0,
    unbiased: bool = True,
) -> np.ndarray:
    """Pairwise CKA across layers.

    Parameters
    ----------
    embeddings:
        ``(n_proteins, n_layers, hidden)`` pooled representations.
    max_samples:
        Subsample this many proteins; CKA is O(n^2) in memory.
    seed:
        Subsampling seed.
    unbiased:
        Use the unbiased HSIC estimator.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n_layers, n_layers)`` CKA matrix.
    """
    n_proteins, n_layers, _ = embeddings.shape
    if n_proteins > max_samples:
        rng = np.random.default_rng(seed)
        index = rng.choice(n_proteins, max_samples, replace=False)
        embeddings = embeddings[np.sort(index)]
        logger.info("Subsampled %d/%d proteins for CKA", max_samples, n_proteins)

    # Gram matrices are computed once per layer rather than once per pair. The
    # naive nested-loop form recomputes them for all L(L-1)/2 pairs -- 561 pairs
    # at 34 layers, i.e. ~30x more O(n^2 d) work than necessary.
    grams: list[np.ndarray] = []
    self_hsic: list[float] = []
    for i in range(n_layers):
        features = np.asarray(embeddings[:, i, :], dtype=np.float64)
        features -= features.mean(axis=0, keepdims=True)
        gram = features @ features.T
        grams.append(gram)
        if unbiased:
            self_hsic.append(unbiased_hsic(gram, gram))
        else:
            centred = _center_gram(gram)
            grams[i] = centred
            self_hsic.append(float(np.sum(centred * centred)))

    matrix = np.eye(n_layers, dtype=np.float64)
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            if unbiased:
                cross = unbiased_hsic(grams[i], grams[j])
            else:
                cross = float(np.sum(grams[i] * grams[j]))
            denominator = np.sqrt(max(self_hsic[i], 0.0) * max(self_hsic[j], 0.0))
            value = float(cross / denominator) if denominator > 0 else float("nan")
            matrix[i, j] = matrix[j, i] = value
    return matrix
