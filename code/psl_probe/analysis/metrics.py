"""Evaluation metrics for multi-label subcellular localization.

Metric choice follows the PSL literature so that probe scores are interpretable
against published supervised baselines:

* **Macro MCC** -- the primary metric. Matthews correlation is well behaved under
  the extreme class imbalance of this task (Peroxisome prevalence is well under
  1%), where F1 and accuracy are dominated by the frequent compartments.
* **Micro / macro F1** -- reported for comparability with DeepLoc-style papers.
* **AUPRC** -- threshold-free, and the appropriate area metric for rare positives.
* **Jaccard (subset) index** -- multi-label set agreement per protein.

All threshold-dependent metrics use a per-class decision threshold tuned on the
validation split, not a fixed 0.5, because a class-balanced BCE objective does not
produce calibrated probabilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise logistic function."""
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-np.clip(logits, -60, None))),
        np.exp(np.clip(logits, None, 60)) / (1.0 + np.exp(np.clip(logits, None, 60))),
    )


def tune_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, n_grid: int = 101
) -> np.ndarray:
    """Per-class thresholds maximising MCC on the given (validation) split."""
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float64)
    grid = np.linspace(0.01, 0.99, n_grid)
    for c in range(n_classes):
        if y_true[:, c].sum() == 0:
            continue
        scores = [matthews_corrcoef(y_true[:, c], (y_score[:, c] >= t).astype(int)) for t in grid]
        thresholds[c] = float(grid[int(np.argmax(scores))])
    return thresholds


def per_class_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Compute per-compartment metrics.

    Classes absent from ``y_true`` yield ``nan`` rather than a misleading zero.
    """
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    rows = []
    for c, name in enumerate(class_names):
        truth = y_true[:, c].astype(int)
        score = y_score[:, c]
        prediction = (score >= thresholds[c]).astype(int)
        n_positive = int(truth.sum())
        if n_positive == 0 or n_positive == len(truth):
            rows.append(
                {
                    "class": name, "n_positive": n_positive, "threshold": float(thresholds[c]),
                    "mcc": np.nan, "f1": np.nan, "auprc": np.nan, "auroc": np.nan,
                }
            )
            continue
        rows.append(
            {
                "class": name,
                "n_positive": n_positive,
                "threshold": float(thresholds[c]),
                "mcc": float(matthews_corrcoef(truth, prediction)),
                "f1": float(f1_score(truth, prediction, zero_division=0)),
                "auprc": float(average_precision_score(truth, score)),
                "auroc": float(roc_auc_score(truth, score)),
            }
        )
    return rows


def multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Aggregate metrics over all compartments."""
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    prediction = (y_score >= thresholds[None, :]).astype(int)
    per_class = per_class_metrics(y_true, y_score, class_names, thresholds)

    mccs = np.array([r["mcc"] for r in per_class], dtype=float)
    auprcs = np.array([r["auprc"] for r in per_class], dtype=float)

    # Jaccard over the predicted label set of each protein; proteins with an empty
    # union (no true and no predicted labels) are scored as perfect agreement.
    intersection = (prediction & y_true.astype(int)).sum(axis=1)
    union = (prediction | y_true.astype(int)).sum(axis=1)
    jaccard = np.where(union == 0, 1.0, intersection / np.maximum(union, 1))

    return {
        "macro_mcc": float(np.nanmean(mccs)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, prediction, average="micro", zero_division=0)),
        "macro_auprc": float(np.nanmean(auprcs)),
        "jaccard": float(jaccard.mean()),
        "subset_accuracy": float((prediction == y_true.astype(int)).all(axis=1).mean()),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    metric: str = "macro_mcc",
    thresholds: np.ndarray | None = None,
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval over proteins.

    Returns ``(point_estimate, lower, upper)``. Resampling is at the protein level;
    because folds are homology-partitioned, proteins in the test split are the
    appropriate independent unit.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = multilabel_metrics(y_true, y_score, class_names, thresholds)[metric]

    values = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        index = rng.integers(0, n, n)
        values[i] = multilabel_metrics(
            y_true[index], y_score[index], class_names, thresholds
        )[metric]

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.nanpercentile(values, [100 * alpha, 100 * (1 - alpha)])
    return float(point), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Paired bootstrap for per-compartment ablation deltas
# ---------------------------------------------------------------------------

def _confusion_codes(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Encode each sample's confusion cell as 0=TN, 1=FP, 2=FN, 3=TP.

    Reducing a prediction to a per-sample categorical code lets a bootstrap
    resample be scored with a single ``bincount`` instead of recomputing the
    metric from scratch, which is what makes a 1000-sample paired bootstrap over
    ten compartments and ten ablation conditions tractable.
    """
    return (2 * y_true.astype(np.int64) + y_pred.astype(np.int64)).astype(np.int64)


def _mcc_from_counts(counts: np.ndarray) -> np.ndarray:
    """Matthews correlation from confusion counts shaped ``(..., 4)``.

    Column order matches :func:`_confusion_codes`: TN, FP, FN, TP. A zero
    denominator (a class with no positives or no predicted positives in the
    resample) yields ``nan`` rather than 0, so it can be excluded from the
    percentile rather than silently biasing it towards zero.
    """
    tn, fp, fn, tp = (counts[..., i].astype(np.float64) for i in range(4))
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    scores_baseline: np.ndarray,
    scores_ablated: np.ndarray,
    thresholds_baseline: np.ndarray,
    thresholds_ablated: np.ndarray,
    class_names: list[str],
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Per-class confidence intervals for the change in MCC under ablation.

    The two probes are evaluated on the *same* test proteins, so the bootstrap is
    paired: each resample draws one set of protein indices and scores both probes
    on it, and the interval is taken over the resulting differences. This is the
    correct estimator for a difference measured on a common test population, and
    it is also the tighter one whenever the two probes succeed and fail on the
    same proteins -- their per-resample MCCs then covary, and that shared variance
    cancels in the difference. (Pairing buys nothing in the degenerate case where
    one probe is perfect and has no resampling variance to cancel.)

    Each probe keeps the per-class decision threshold tuned on its own validation
    split, so the comparison is between operating points that were each selected
    without reference to the test split.

    Parameters
    ----------
    y_true:
        ``(n_proteins, n_classes)`` binary test labels.
    scores_baseline, scores_ablated:
        ``(n_proteins, n_classes)`` predicted probabilities.
    thresholds_baseline, thresholds_ablated:
        Per-class decision thresholds for each probe.
    class_names:
        Names in label-column order.
    n_samples:
        Bootstrap resamples.
    confidence:
        Interval coverage.
    seed:
        Seed for the resampling.

    Returns
    -------
    list[dict]
        One record per class with ``delta`` (point estimate), ``delta_lo``,
        ``delta_hi``, and ``frac_negative`` -- the fraction of resamples in which
        the delta was below zero, which serves as a one-sided significance
        summary for a degradation.
    """
    n_proteins, n_classes = y_true.shape
    truth = y_true.astype(np.int64)
    pred_baseline = (scores_baseline >= thresholds_baseline[None, :]).astype(np.int64)
    pred_ablated = (scores_ablated >= thresholds_ablated[None, :]).astype(np.int64)

    codes_baseline = _confusion_codes(truth, pred_baseline)   # (n, C)
    codes_ablated = _confusion_codes(truth, pred_ablated)

    # Offset each class into its own 4-slot block so one bincount scores them all.
    offsets = 4 * np.arange(n_classes, dtype=np.int64)[None, :]
    flat_baseline = codes_baseline + offsets
    flat_ablated = codes_ablated + offsets

    point_baseline = _mcc_from_counts(
        np.bincount(flat_baseline.ravel(), minlength=4 * n_classes).reshape(n_classes, 4)
    )
    point_ablated = _mcc_from_counts(
        np.bincount(flat_ablated.ravel(), minlength=4 * n_classes).reshape(n_classes, 4)
    )

    deltas = _bootstrap_delta_draws(
        flat_baseline, flat_ablated, n_classes, n_proteins, n_samples, seed
    )

    alpha = (1.0 - confidence) / 2.0
    lower = np.nanpercentile(deltas, 100 * alpha, axis=0)
    upper = np.nanpercentile(deltas, 100 * (1 - alpha), axis=0)

    records = []
    for c, name in enumerate(class_names):
        column = deltas[:, c]
        finite = column[np.isfinite(column)]
        records.append(
            {
                "class": name,
                "delta": float(point_ablated[c] - point_baseline[c]),
                "delta_lo": float(lower[c]),
                "delta_hi": float(upper[c]),
                "frac_negative": float((finite < 0).mean()) if finite.size else float("nan"),
                "n_bootstrap": n_samples,
            }
        )
    return records


def _bootstrap_mcc_draws(
    flat_reference: np.ndarray,
    flat_comparison: np.ndarray,
    n_classes: int,
    n_proteins: int,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired bootstrap draws of both probes' per-class MCC.

    Returns ``(reference, comparison)``, each ``(n_samples, n_classes)``. Both are
    returned rather than only their difference so that a caller can also form a
    *ratio* on each resample. The N-terminal retention statistic is a ratio, and
    the ratio of two separately bootstrapped quantities is not the bootstrap of
    the ratio: only a shared resample preserves the correlation between numerator
    and denominator.
    """
    rng = np.random.default_rng(seed)
    reference = np.empty((n_samples, n_classes), dtype=np.float64)
    comparison = np.empty((n_samples, n_classes), dtype=np.float64)
    for b in range(n_samples):
        index = rng.integers(0, n_proteins, n_proteins)   # shared by both probes
        reference[b] = _mcc_from_counts(
            np.bincount(flat_reference[index].ravel(), minlength=4 * n_classes)
            .reshape(n_classes, 4)
        )
        comparison[b] = _mcc_from_counts(
            np.bincount(flat_comparison[index].ravel(), minlength=4 * n_classes)
            .reshape(n_classes, 4)
        )
    return reference, comparison


def _bootstrap_delta_draws(
    flat_baseline: np.ndarray,
    flat_ablated: np.ndarray,
    n_classes: int,
    n_proteins: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Paired bootstrap draws of the per-class MCC difference."""
    reference, comparison = _bootstrap_mcc_draws(
        flat_baseline, flat_ablated, n_classes, n_proteins, n_samples, seed
    )
    return comparison - reference


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values.

    Preferred over Bonferroni for the per-compartment family here: with ten
    compartments, Bonferroni is conservative enough to discard effects the data
    genuinely support, whereas controlling the false discovery rate is the
    conventional choice when the tests are exploratory and partly correlated.
    Bonferroni-adjusted simultaneous intervals are reported alongside, since an
    interval and a q-value answer different questions.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = pvalues.size
    order = np.argsort(pvalues)
    ranks = np.arange(1, n + 1)
    adjusted = pvalues[order] * n / ranks
    # Enforce monotonicity from the largest p downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def multiseed_paired_bootstrap_deltas(
    labels: np.ndarray,
    baseline_scores: list[np.ndarray],
    ablated_scores: list[np.ndarray],
    baseline_thresholds: list[np.ndarray],
    ablated_thresholds: list[np.ndarray],
    class_names: list[str],
    n_samples_per_seed: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    include_ratio: bool = False,
) -> list[dict]:
    """Per-class deltas with both variance components and multiplicity control.

    Two things vary between one reported delta and another: which proteins happen
    to be in the test split, and how the probe happened to initialise and shuffle
    during fitting. A bootstrap alone captures only the first, and refitting under
    several seeds alone captures only the second. Here each seed contributes its
    own paired bootstrap and the draws are **pooled** across seeds, so the
    resulting interval marginalises over both -- a mixture over the seed
    distribution rather than a conditional-on-one-fit interval.

    The pooled draws also give a two-sided bootstrap p-value per compartment,
    which is corrected across the compartment family two ways: Benjamini-Hochberg
    q-values, and Bonferroni-adjusted *simultaneous* intervals at level
    ``1 - (1 - confidence) / n_classes``.

    Parameters
    ----------
    labels:
        ``(n_proteins, n_classes)`` binary test labels, shared across seeds
        (the homology split is fixed; only the probe fit varies).
    baseline_scores, ablated_scores:
        One ``(n_proteins, n_classes)`` score matrix per seed.
    baseline_thresholds, ablated_thresholds:
        One threshold vector per seed; each probe keeps the operating point tuned
        on its own validation split.
    class_names:
        Names in label-column order.
    n_samples_per_seed:
        Bootstrap resamples drawn within each seed.
    confidence:
        Nominal coverage of the per-compartment interval.
    seed:
        Base seed for resampling; each fit-seed uses a distinct stream.

    Returns
    -------
    list[dict]
        Per class: ``delta`` (mean over seeds), ``delta_lo``/``delta_hi``
        (pooled, unadjusted), ``delta_lo_adj``/``delta_hi_adj`` (Bonferroni
        simultaneous), ``seed_sd`` (between-seed standard deviation of the point estimate),
        ``p_bootstrap``, ``q_bh``, ``n_seeds`` and ``n_bootstrap``.
    """
    n_seeds = len(baseline_scores)
    if not (len(ablated_scores) == len(baseline_thresholds) == len(ablated_thresholds) == n_seeds):
        raise ValueError("baseline/ablated scores and thresholds must have one entry per seed")
    if n_seeds == 0:
        raise ValueError("at least one seed is required")

    n_proteins, n_classes = labels.shape
    truth = labels.astype(np.int64)
    offsets = 4 * np.arange(n_classes, dtype=np.int64)[None, :]

    per_seed_point = np.empty((n_seeds, n_classes), dtype=np.float64)
    per_seed_reference = np.empty((n_seeds, n_classes), dtype=np.float64)
    per_seed_comparison = np.empty((n_seeds, n_classes), dtype=np.float64)
    pooled: list[np.ndarray] = []
    pooled_ref: list[np.ndarray] = []
    pooled_cmp: list[np.ndarray] = []

    for s in range(n_seeds):
        flat_b = _confusion_codes(
            truth, (baseline_scores[s] >= baseline_thresholds[s][None, :]).astype(np.int64)
        ) + offsets
        flat_a = _confusion_codes(
            truth, (ablated_scores[s] >= ablated_thresholds[s][None, :]).astype(np.int64)
        ) + offsets

        counts_b = np.bincount(flat_b.ravel(), minlength=4 * n_classes).reshape(n_classes, 4)
        counts_a = np.bincount(flat_a.ravel(), minlength=4 * n_classes).reshape(n_classes, 4)
        per_seed_reference[s] = _mcc_from_counts(counts_b)
        per_seed_comparison[s] = _mcc_from_counts(counts_a)
        per_seed_point[s] = per_seed_comparison[s] - per_seed_reference[s]

        ref_draws, cmp_draws = _bootstrap_mcc_draws(
            flat_b, flat_a, n_classes, n_proteins, n_samples_per_seed, seed + s
        )
        pooled.append(cmp_draws - ref_draws)
        pooled_ref.append(ref_draws)
        pooled_cmp.append(cmp_draws)

    draws = np.concatenate(pooled, axis=0)          # (n_seeds * n_samples, n_classes)

    alpha = 1.0 - confidence
    lower = np.nanpercentile(draws, 100 * (alpha / 2), axis=0)
    upper = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)

    # Bonferroni-adjusted simultaneous interval across the compartment family.
    alpha_adj = alpha / n_classes
    lower_adj = np.nanpercentile(draws, 100 * (alpha_adj / 2), axis=0)
    upper_adj = np.nanpercentile(draws, 100 * (1 - alpha_adj / 2), axis=0)

    pvalues = np.empty(n_classes, dtype=np.float64)
    for c in range(n_classes):
        column = draws[:, c]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            pvalues[c] = np.nan
            continue
        tail = min((finite <= 0).mean(), (finite >= 0).mean())
        # Floor at 1/B: a bootstrap cannot resolve a p-value below its resolution.
        pvalues[c] = float(np.clip(2 * tail, 1.0 / finite.size, 1.0))

    qvalues = benjamini_hochberg(pvalues)

    ratio_fields: list[dict] = [{} for _ in range(n_classes)]
    if include_ratio:
        ref_draws = np.concatenate(pooled_ref, axis=0)
        cmp_draws = np.concatenate(pooled_cmp, axis=0)
        # A ratio is undefined where the reference MCC is at or below zero; those
        # draws are dropped rather than allowed to produce a spurious extreme.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_draws = np.where(ref_draws > 0, cmp_draws / ref_draws, np.nan)
        for c in range(n_classes):
            column = ratio_draws[:, c]
            finite = column[np.isfinite(column)]
            ratio_fields[c] = {
                "ratio": float(
                    np.nanmean(
                        np.where(
                            per_seed_reference[:, c] > 0,
                            per_seed_comparison[:, c] / per_seed_reference[:, c],
                            np.nan,
                        )
                    )
                ),
                "ratio_lo": float(np.nanpercentile(finite, 100 * (alpha / 2)))
                if finite.size else float("nan"),
                "ratio_hi": float(np.nanpercentile(finite, 100 * (1 - alpha / 2)))
                if finite.size else float("nan"),
                "frac_ratio_undefined": float(1.0 - finite.size / column.size),
            }

    return [
        {
            "class": name,
            "reference_mcc": float(np.nanmean(per_seed_reference[:, c])),
            "comparison_mcc": float(np.nanmean(per_seed_comparison[:, c])),
            "delta": float(np.nanmean(per_seed_point[:, c])),
            "delta_lo": float(lower[c]),
            "delta_hi": float(upper[c]),
            "delta_lo_adj": float(lower_adj[c]),
            "delta_hi_adj": float(upper_adj[c]),
            "seed_sd": float(np.nanstd(per_seed_point[:, c], ddof=1)) if n_seeds > 1 else 0.0,
            "p_bootstrap": float(pvalues[c]),
            "q_bh": float(qvalues[c]),
            "n_seeds": n_seeds,
            "n_bootstrap": int(draws.shape[0]),
            **ratio_fields[c],
        }
        for c, name in enumerate(class_names)
    ]
