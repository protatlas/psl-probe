"""Publication figure style and reusable panels.

All figures are written as vector PDF (for the manuscript) and 300 dpi PNG (for
quick inspection). The style targets a single-column width of 3.5 in and a
double-column width of 7.2 in, which matches most computational-biology journals,
so figures are never rescaled in LaTeX and font sizes stay consistent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#: Figure widths in inches.
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.2

#: Colour-blind-safe qualitative palette (Okabe & Ito, 2008).
OKABE_ITO = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
)


def set_style(font_size: float = 8.0) -> None:
    """Apply the manuscript-wide Matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": font_size,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO)),
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": font_size - 1,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, path: Path | str, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Write a figure in every requested format and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        fig.savefig(target, format=fmt)
        written.append(target)
    plt.close(fig)
    return written


def annotate_heatmap(
    ax: plt.Axes, matrix: np.ndarray, fmt: str = "{:.2f}", threshold: float | None = None
) -> None:
    """Write cell values onto a heatmap, switching text colour for contrast."""
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    cutoff = threshold if threshold is not None else float(np.nanmean(finite))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            ax.text(
                j, i, fmt.format(value),
                ha="center", va="center",
                color="white" if value > cutoff else "black",
                fontsize=mpl.rcParams["font.size"] - 2.5,
            )


def add_panel_labels(
    axes, labels: str = "abcdefgh", dx: float = -34.0, dy: float = 8.0
) -> None:
    """Add bold lower-case panel labels above and left of each axes.

    The offset is given in *typographic points* rather than axes fractions: an
    axes-fraction offset scales with panel width, so on a wide three-panel figure
    the label drifts inwards and collides with a centred title. A fixed point
    offset keeps every label the same distance from its panel corner.
    """
    for ax, label in zip(np.ravel(axes), labels, strict=False):
        ax.annotate(
            label,
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(dx, dy), textcoords="offset points",
            fontsize=mpl.rcParams["font.size"] + 1, fontweight="bold",
            va="bottom", ha="left", annotation_clip=False,
        )


def tidy_log_axis(ax: plt.Axes, which: str = "x") -> None:
    """Make a log axis readable at figure font sizes.

    Two failure modes have to be handled together:

    * Over a wide range, Matplotlib labels the minor ticks as well
      (2x10^2, 3x10^2, ...) and they overlap. Suppress them.
    * Over a range narrower than one decade there may be no major tick inside the
      view at all, so suppressing the minor labels would leave the axis with no
      labels whatsoever. Label a thinned set of minor ticks with plain numbers
      instead.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    low, high = axis.get_view_interval()
    if low <= 0 or high <= 0:
        return

    if np.log10(high) - np.log10(low) >= 1.0:
        axis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
    else:
        axis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
        axis.set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.set_minor_formatter(mpl.ticker.ScalarFormatter())
