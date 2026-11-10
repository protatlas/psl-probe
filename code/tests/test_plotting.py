"""Figure-style helpers must produce labelled, non-colliding axes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from psl_probe.analysis.plotting import (  # noqa: E402
    add_panel_labels,
    save_figure,
    set_style,
    tidy_log_axis,
)


def _tick_labels(axis) -> list[str]:
    """All non-empty tick label strings currently on an axis."""
    labels = [t.get_text() for t in axis.get_ticklabels(which="both")]
    return [t for t in labels if t.strip()]


def test_wide_log_range_keeps_major_labels_only() -> None:
    set_style()
    fig, ax = plt.subplots()
    ax.plot([1, 10, 100, 10000], [1, 2, 3, 4])
    ax.set_xscale("log")
    fig.canvas.draw()
    tidy_log_axis(ax, "x")
    fig.canvas.draw()
    assert len(_tick_labels(ax.xaxis)) > 0
    plt.close(fig)


def test_narrow_log_range_still_gets_labels() -> None:
    """A sub-decade log range has no major tick in view; it must not end up bare.

    This is the failure the helper exists to prevent: suppressing minor labels on
    a 170-210 range leaves an axis with no numbers on it at all.
    """
    set_style()
    fig, ax = plt.subplots()
    ax.bar(range(10), np.linspace(210, 165, 10))
    ax.set_yscale("log")
    fig.canvas.draw()
    tidy_log_axis(ax, "y")
    fig.canvas.draw()
    assert len(_tick_labels(ax.yaxis)) > 0
    plt.close(fig)


def test_non_positive_range_is_ignored_not_crashed() -> None:
    set_style()
    fig, ax = plt.subplots()
    ax.plot([-5, 0, 5], [1, 2, 3])
    tidy_log_axis(ax, "x")  # linear axis: must be a no-op, not an exception
    plt.close(fig)


def test_panel_labels_sit_outside_the_axes() -> None:
    """Point-offset labels must land left of and above the axes box."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
    add_panel_labels(axes)
    fig.canvas.draw()
    for ax, expected in zip(np.ravel(axes), "abc", strict=False):
        annotations = [c for c in ax.texts if c.get_text() == expected]
        assert len(annotations) == 1
        window = annotations[0].get_window_extent(fig.canvas.get_renderer())
        box = ax.get_window_extent()
        assert window.x1 <= box.x0 + 1  # left of the axes
        assert window.y0 >= box.y1 - 1  # above the axes
    plt.close(fig)


def test_save_figure_writes_both_formats(tmp_path) -> None:
    set_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save_figure(fig, tmp_path / "demo")
    assert {p.suffix for p in written} == {".pdf", ".png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
