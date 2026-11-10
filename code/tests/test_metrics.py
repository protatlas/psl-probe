"""Metric behaviour under the imbalance regime of this benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from psl_probe.analysis.metrics import (
    bootstrap_ci,
    multilabel_metrics,
    per_class_metrics,
    sigmoid,
    tune_thresholds,
)

CLASSES = ["a", "b", "c"]


def test_perfect_prediction_scores_one() -> None:
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    metrics = multilabel_metrics(y, y.astype(float), CLASSES)
    assert metrics["macro_mcc"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["subset_accuracy"] == 1.0
    assert metrics["jaccard"] == 1.0


def test_absent_class_is_nan_not_zero() -> None:
    """A class with no positives is undefined, not perfectly wrong."""
    y = np.zeros((10, 3))
    y[:5, 0] = 1
    rows = per_class_metrics(y, np.random.default_rng(0).random((10, 3)), CLASSES)
    assert np.isnan(rows[1]["mcc"])
    assert rows[1]["n_positive"] == 0


def test_sigmoid_is_stable_at_extremes() -> None:
    values = sigmoid(np.array([-1e4, -50.0, 0.0, 50.0, 1e4]))
    assert np.all(np.isfinite(values))
    assert values[0] == 0.0
    assert values[2] == 0.5
    assert values[-1] == 1.0


def test_threshold_tuning_beats_fixed_half_under_imbalance() -> None:
    """With a rare positive class, a tuned threshold must not be worse than 0.5."""
    rng = np.random.default_rng(0)
    n = 2000
    y = np.zeros((n, 3))
    y[: int(0.02 * n), 0] = 1  # 2% prevalence
    y[: int(0.30 * n), 1] = 1
    y[: int(0.50 * n), 2] = 1
    # Scores correlated with truth but poorly calibrated (compressed towards zero).
    score = np.clip(0.3 * y + 0.15 * rng.random((n, 3)), 0, 1)

    tuned = tune_thresholds(y, score)
    fixed = multilabel_metrics(y, score, CLASSES)["macro_mcc"]
    improved = multilabel_metrics(y, score, CLASSES, tuned)["macro_mcc"]
    assert improved >= fixed


def test_bootstrap_interval_brackets_point_estimate() -> None:
    """The interval must contain the point estimate and have non-zero width.

    The scores below deliberately overlap between classes; a perfectly separable
    synthetic task would score MCC = 1 on every resample and yield a degenerate
    zero-width interval that tests nothing.
    """
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random((n, 3)) < 0.3).astype(float)
    score = np.clip(0.25 * y + 0.75 * rng.random((n, 3)), 0, 1)
    point, lower, upper = bootstrap_ci(y, score, CLASSES, n_samples=200, seed=0)
    assert lower <= point <= upper
    assert upper - lower > 0


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _correlated_probe_pair(n=3000, n_classes=3, seed=0):
    """Two imperfect probes that share a per-protein difficulty term.

    This mirrors the real baseline/ablated comparison: the same proteins are hard
    for both probes. A construction where either probe is perfect has no
    resampling variance and cannot exercise the pairing logic.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random((n, n_classes)) < np.array([0.4, 0.15, 0.05])).astype(float)
    latent = 2 * y - 1
    difficulty = rng.normal(0, 1.6, (n, n_classes))
    strong = _sigmoid(1.8 * latent + difficulty)
    weak = _sigmoid(0.9 * latent + difficulty)
    return y, strong, weak


def test_fast_mcc_matches_sklearn() -> None:
    from psl_probe.analysis.metrics import _confusion_codes, _mcc_from_counts

    y, scores, _ = _correlated_probe_pair()
    n_classes = y.shape[1]
    thresholds = np.full(n_classes, 0.5)
    codes = _confusion_codes(y.astype(int), (scores >= thresholds).astype(int))
    offsets = 4 * np.arange(n_classes)[None, :]
    counts = np.bincount((codes + offsets).ravel(), minlength=4 * n_classes).reshape(n_classes, 4)
    fast = _mcc_from_counts(counts)
    reference = np.array(
        [matthews_corrcoef(y[:, c], (scores[:, c] >= 0.5).astype(int)) for c in range(n_classes)]
    )
    assert np.allclose(fast, reference, atol=1e-12)


def test_paired_interval_brackets_point_estimate() -> None:
    from psl_probe.analysis.metrics import paired_bootstrap_deltas

    y, strong, weak = _correlated_probe_pair()
    names = ["a", "b", "c"]
    thresholds = np.full(3, 0.5)
    rows = paired_bootstrap_deltas(y, strong, weak, thresholds, thresholds, names, n_samples=300)
    for row in rows:
        assert row["delta_lo"] <= row["delta"] <= row["delta_hi"]
        assert row["delta"] < 0  # the weaker probe must score worse
        assert 0.0 <= row["frac_negative"] <= 1.0


def test_pairing_is_tighter_than_unpaired_when_probes_covary() -> None:
    """The variance reduction that justifies pairing, measured directly."""
    from psl_probe.analysis.metrics import (
        _confusion_codes,
        _mcc_from_counts,
        paired_bootstrap_deltas,
    )

    y, strong, weak = _correlated_probe_pair()
    n, n_classes = y.shape
    thresholds = np.full(n_classes, 0.5)
    names = [f"c{i}" for i in range(n_classes)]

    paired = paired_bootstrap_deltas(
        y, strong, weak, thresholds, thresholds, names, n_samples=600, seed=0
    )
    paired_width = np.array([r["delta_hi"] - r["delta_lo"] for r in paired])

    offsets = 4 * np.arange(n_classes)[None, :]
    codes_a = _confusion_codes(y.astype(int), (strong >= thresholds).astype(int)) + offsets
    codes_b = _confusion_codes(y.astype(int), (weak >= thresholds).astype(int)) + offsets
    rng = np.random.default_rng(0)
    unpaired = np.empty((600, n_classes))
    for b in range(600):
        ia, ib = rng.integers(0, n, n), rng.integers(0, n, n)
        unpaired[b] = _mcc_from_counts(
            np.bincount(codes_b[ib].ravel(), minlength=4 * n_classes).reshape(n_classes, 4)
        ) - _mcc_from_counts(
            np.bincount(codes_a[ia].ravel(), minlength=4 * n_classes).reshape(n_classes, 4)
        )
    unpaired_width = np.nanpercentile(unpaired, 97.5, axis=0) - np.nanpercentile(
        unpaired, 2.5, axis=0
    )
    assert (unpaired_width > paired_width).all()


def test_identical_probes_give_zero_delta() -> None:
    """A probe compared against itself must have delta 0 and a degenerate interval."""
    from psl_probe.analysis.metrics import paired_bootstrap_deltas

    y, scores, _ = _correlated_probe_pair()
    thresholds = np.full(3, 0.5)
    rows = paired_bootstrap_deltas(
        y, scores, scores, thresholds, thresholds, ["a", "b", "c"], n_samples=200
    )
    for row in rows:
        assert row["delta"] == 0.0
        assert row["delta_lo"] == 0.0 and row["delta_hi"] == 0.0


def test_benjamini_hochberg_matches_reference() -> None:
    """BH q-values against a known worked example, plus the shape guarantees."""
    from psl_probe.analysis.metrics import benjamini_hochberg

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = benjamini_hochberg(p)
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(q >= p - 1e-12)              # adjustment never decreases a p-value
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)  # monotone in p
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return
    assert np.allclose(q, multipletests(p, method="fdr_bh")[1])


def test_multiseed_widens_intervals_and_reports_seed_spread() -> None:
    """Pooling across seeds must add the fitting-variance component."""
    from psl_probe.analysis.metrics import multiseed_paired_bootstrap_deltas

    rng = np.random.default_rng(0)
    y, strong, weak = _correlated_probe_pair(n=2500, seed=3)
    names = ["a", "b", "c"]
    thr = [np.full(3, 0.5) for _ in range(4)]
    # Four "fits" that differ slightly, as refitting under different seeds would.
    strongs = [np.clip(strong + rng.normal(0, 0.03, strong.shape), 0, 1) for _ in range(4)]
    weaks = [np.clip(weak + rng.normal(0, 0.03, weak.shape), 0, 1) for _ in range(4)]

    single = multiseed_paired_bootstrap_deltas(
        y, strongs[:1], weaks[:1], thr[:1], thr[:1], names, 400, seed=0
    )
    multi = multiseed_paired_bootstrap_deltas(
        y, strongs, weaks, thr, thr, names, 400, seed=0
    )
    for one, many in zip(single, multi, strict=True):
        assert one["seed_sd"] == 0.0 and many["seed_sd"] > 0.0
        assert many["n_seeds"] == 4
        # Bonferroni simultaneous intervals are never narrower than unadjusted.
        assert many["delta_lo_adj"] <= many["delta_lo"]
        assert many["delta_hi_adj"] >= many["delta_hi"]
        assert 0.0 <= many["q_bh"] <= 1.0


def test_multiseed_rejects_ragged_input() -> None:
    from psl_probe.analysis.metrics import multiseed_paired_bootstrap_deltas

    y, strong, weak = _correlated_probe_pair(n=500)
    thr = np.full(3, 0.5)
    with pytest.raises(ValueError, match="one entry per seed"):
        multiseed_paired_bootstrap_deltas(
            y, [strong, strong], [weak], [thr, thr], [thr], ["a", "b", "c"]
        )
