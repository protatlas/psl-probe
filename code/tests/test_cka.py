"""CKA must satisfy its defining invariances."""

from __future__ import annotations

import numpy as np
import pytest

from psl_probe.analysis.cka import layer_cka_matrix, linear_cka


def test_identity_is_one() -> None:
    x = np.random.default_rng(0).normal(size=(200, 16))
    assert linear_cka(x, x) == pytest.approx(1.0)


def test_invariant_to_rotation_and_isotropic_scaling() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16))
    rotation, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    assert linear_cka(x, 7.5 * (x @ rotation)) == pytest.approx(1.0)


def test_independent_representations_score_near_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 16))
    y = rng.normal(size=(500, 16))
    assert abs(linear_cka(x, y)) < 0.05


def test_layer_matrix_is_symmetric_with_unit_diagonal() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(120, 5, 8))
    matrix = layer_cka_matrix(embeddings, max_samples=120)
    assert matrix.shape == (5, 5)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
