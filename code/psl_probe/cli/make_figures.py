"""Stage 5: render every manuscript figure from the stored result files.

Each figure is generated only when its inputs exist, so the script can be re-run
at any point in the pipeline; missing inputs are reported and skipped rather than
raising. Outputs land in ``visualization/figures`` as paired PDF/PNG.

Example
-------
::

    python -m psl_probe.cli.make_figures --config configs/esm2_650m.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psl_probe.analysis.plotting import (
    DOUBLE_COLUMN,
    OKABE_ITO,
    SINGLE_COLUMN,
    add_panel_labels,
    annotate_heatmap,
    save_figure,
    set_style,
    tidy_log_axis,
)
from psl_probe.cli import add_common_arguments, resolve
from psl_probe.data.labels import CLASS_ABBREVIATIONS, LOCALIZATION_CLASSES
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

#: Model tags are filesystem identifiers; figures must show the published model
#: name instead. An unknown tag falls back to the tag itself rather than raising,
#: so adding a model never breaks figure generation.
MODEL_DISPLAY_NAMES = {
    "esm2_35m": "ESM-2 35M",
    "esm2_150m": "ESM-2 150M",
    "esm2_650m": "ESM-2 650M",
    "prott5_xl": "ProtT5-XL-U50",
}


def display_name(tag: str) -> str:
    """Return the published name for a model tag, e.g. ``esm2_650m`` -> ESM-2 650M."""
    return MODEL_DISPLAY_NAMES.get(tag, tag)


#: Sorting regime per compartment, mirroring Section 2.4 of the manuscript.
#: 1 = cleavable N-terminal presequence, 2 = discrete internal or C-terminal
#: motif, 3 = no linear sequence determinant.
#:
#: One approximation: regime 2 describes the *soluble KDEL-bearing residents* of
#: the ER, whereas the benchmark class covers all ER proteins including membrane
#: ones, whose determinants are distributed. The ER row therefore mixes two
#: regimes and should not be read as a clean test.
REGIME = {
    "Extracellular": 1, "Mitochondrion": 1, "Plastid": 1,
    "Nucleus": 2, "Endoplasmic reticulum": 2, "Peroxisome": 2,
    "Cell membrane": 3, "Golgi apparatus": 3, "Lysosome/Vacuole": 3,
    "Cytoplasm": 3,
}

#: Regime is ordinal -- it ranks how discrete and positionally fixed a
#: compartment's determinant is -- so it is encoded as a single-hue sequential
#: ramp rather than as unrelated categorical hues. A ramp is also the safest
#: choice under colour-vision deficiency: the previous blue/light-blue/grey set
#: left its two lightest classes at dE 12.8 for normal vision and 11.3 under
#: deuteranopia, below the dE 15 threshold at which a pair is reliably
#: separable. This ramp measures 19.1 and 17.8. The lightest step sits below 3:1
#: against white, so bars drawn in it carry a thin dark edge and a direct label.
REGIME_COLOURS = {1: "#08519C", 2: "#4292C6", 3: "#9ECAE1"}
REGIME_EDGE = "#08306B"
REGIME_LABELS = {
    1: "Regime 1: cleavable N-terminal presequence",
    2: "Regime 2: signal internal or C-terminal",
    3: "Regime 3: no linear sequence determinant",
}


def figure_dataset_overview(cfg) -> Path | None:
    """Figure 1: class prevalence, label co-occurrence and length distribution."""
    benchmark_path = cfg.paths.processed / "benchmark.parquet"
    if not benchmark_path.exists():
        logger.warning("Skipping dataset overview: %s missing", benchmark_path)
        return None
    dataset = pd.read_parquet(benchmark_path)
    classes = list(LOCALIZATION_CLASSES)
    abbreviations = [CLASS_ABBREVIATIONS[c] for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.3))

    counts = dataset[classes].sum().to_numpy()
    order = np.argsort(-counts)
    axes[0].bar(range(len(classes)), counts[order], color="#0072B2")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels([abbreviations[i] for i in order], rotation=60, ha="right")
    axes[0].set_ylabel("Proteins")
    axes[0].set_yscale("log")
    tidy_log_axis(axes[0], "y")
    axes[0].set_title("Compartment prevalence")

    labels = dataset[classes].to_numpy(dtype=float)
    # Jaccard co-occurrence: |A and B| / |A or B|, symmetric and prevalence-robust.
    intersection = labels.T @ labels
    totals = labels.sum(axis=0)
    union = totals[:, None] + totals[None, :] - intersection
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    np.fill_diagonal(jaccard, np.nan)
    image = axes[1].imshow(jaccard, cmap="magma", vmin=0)
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(abbreviations, rotation=60, ha="right")
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_yticklabels(abbreviations)
    axes[1].set_title("Label co-occurrence (Jaccard)")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].hist(dataset["length"], bins=60, color="#009E73")
    axes[2].set_xlabel("Sequence length (residues)")
    axes[2].set_ylabel("Proteins")
    axes[2].set_xscale("log")
    tidy_log_axis(axes[2], "x")
    axes[2].set_title(f"Length distribution (n={len(dataset):,})")

    add_panel_labels(axes)
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig1_dataset_overview")[0]


def figure_layer_profile(cfg) -> Path | None:
    """Figure 2: layer-wise probe performance, control task and selectivity."""
    path = cfg.paths.probes / f"{cfg.model.tag}_probe_results.csv"
    if not path.exists():
        logger.warning("Skipping layer profile: %s missing", path)
        return None
    results = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.4), sharex=True)
    for pooling, group in results[results["task"] == "real"].groupby("pooling"):
        # Average over fit seeds where several are present; shade +/- 1 s.d.
        stats = (
            group.groupby("layer")["macro_mcc"].agg(["mean", "std", "size"]).reset_index()
        )
        line, = axes[0].plot(
            stats["layer"], stats["mean"], marker="o", label=f"{pooling} (real)"
        )
        if stats["size"].max() > 1:
            axes[0].fill_between(
                stats["layer"],
                stats["mean"] - stats["std"].fillna(0),
                stats["mean"] + stats["std"].fillna(0),
                alpha=0.25, linewidth=0, color=line.get_color(),
            )
    for pooling, group in results[results["task"] == "control"].groupby("pooling"):
        aggregated = group.groupby("layer")["macro_mcc"].agg(["mean", "std"]).reset_index()
        axes[0].plot(
            aggregated["layer"], aggregated["mean"], linestyle="--", marker=None,
            label=f"{pooling} (control)",
        )
        axes[0].fill_between(
            aggregated["layer"],
            aggregated["mean"] - aggregated["std"].fillna(0),
            aggregated["mean"] + aggregated["std"].fillna(0),
            alpha=0.15, linewidth=0,
        )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Macro MCC")
    axes[0].set_title(f"{display_name(cfg.model.tag)}: layer-wise decodability")
    axes[0].legend(ncol=2)

    selectivity_path = cfg.paths.probes / f"{cfg.model.tag}_selectivity.csv"
    if selectivity_path.exists():
        selectivity = pd.read_csv(selectivity_path)
        for pooling, group in selectivity.groupby("pooling"):
            group = group.sort_values("layer")
            line, = axes[1].plot(
                group["layer"], group["selectivity"], marker="o", label=pooling
            )
            if "selectivity_sd" in group and group["selectivity_sd"].notna().any():
                axes[1].fill_between(
                    group["layer"],
                    group["selectivity"] - group["selectivity_sd"].fillna(0),
                    group["selectivity"] + group["selectivity_sd"].fillna(0),
                    alpha=0.25, linewidth=0, color=line.get_color(),
                )
        axes[1].axhline(0.0, color="0.6", linewidth=0.6, linestyle=":")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Selectivity (real $-$ control)")
        axes[1].set_title("Probe selectivity")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    add_panel_labels(axes)
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig2_layer_profile")[0]


def figure_per_class_layers(cfg) -> Path | None:
    """Figure 3: per-compartment MCC as a function of layer depth."""
    path = cfg.paths.probes / f"{cfg.model.tag}_per_class.csv"
    if not path.exists():
        logger.warning("Skipping per-class layer heatmap: %s missing", path)
        return None
    per_class = pd.read_csv(path)
    strategy = per_class["pooling"].iloc[0]
    subset = per_class[per_class["pooling"] == strategy]

    pivot = subset.pivot_table(index="class", columns="layer", values="mcc")
    pivot = pivot.reindex([c for c in LOCALIZATION_CLASSES if c in pivot.index])

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 2.6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([CLASS_ABBREVIATIONS.get(c, c) for c in pivot.index])
    ax.set_xticks(range(0, pivot.shape[1], max(pivot.shape[1] // 12, 1)))
    ax.set_xticklabels(pivot.columns[:: max(pivot.shape[1] // 12, 1)])
    ax.set_xlabel("Layer")
    ax.set_title(f"{display_name(cfg.model.tag)} ({strategy} pooling): per-compartment MCC")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="MCC")
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig3_per_class_layers")[0]


def figure_cka(cfg, strategy: str = "mean") -> Path | None:
    """Figure 4: layer-by-layer representational similarity."""
    path = cfg.paths.analysis / f"{cfg.model.tag}_{strategy}_cka.npy"
    if not path.exists():
        logger.warning("Skipping CKA figure: %s missing", path)
        return None
    matrix = np.load(path)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN))
    image = ax.imshow(matrix, cmap="rocket" if "rocket" in plt.colormaps() else "magma",
                      vmin=0, vmax=1, origin="lower")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"{display_name(cfg.model.tag)}: linear CKA")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="CKA")
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig4_cka")[0]


def figure_attention(cfg) -> Path | None:
    """Figure 5: per-head attention enrichment on sorting signals vs a decoy null."""
    path = cfg.paths.analysis / f"{cfg.model.tag}_attention.npz"
    if not path.exists():
        logger.warning("Skipping attention figure: %s missing", path)
        return None
    payload = np.load(path)
    signal, decoy = payload["signal_mean"], payload["decoy_mean"]

    has_excess = "excess_mean" in payload
    n_panels = 3 if has_excess else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(DOUBLE_COLUMN, 2.6))

    vmax = float(np.nanmax([np.nanmax(signal), np.nanmax(decoy)]))
    # Attention exists only for transformer blocks, so the stored array has one
    # row per block. The rest of the paper indexes hidden states, where state b
    # is the output of block b and state 0 is the input embedding. Shifting the
    # axis to 1..n_blocks makes a head's layer number mean the same thing here as
    # it does in every other figure and in the text.
    n_blocks, n_heads = signal.shape
    extent = (0.5, n_blocks + 0.5, -0.5, n_heads - 0.5)
    for ax, matrix, title in ((axes[0], signal, "Curated SIGNAL/TRANSIT"),
                              (axes[1], decoy, "Length-matched decoy")):
        image = ax.imshow(matrix.T, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                          origin="lower", extent=extent)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Enrichment")

    if has_excess:
        # Paired signal-minus-decoy excess, diverging about zero: blue marks heads
        # that avoid signal regions, red those that seek them.
        excess = payload["excess_mean"]
        limit = float(np.nanmax(np.abs(excess)))
        ax = axes[2]
        image = ax.imshow(
            excess.T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
            origin="lower", extent=extent,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        ax.set_title("Paired excess (signal $-$ decoy)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=r"$\Delta$ enrichment")

    add_panel_labels(axes)
    fig.suptitle(
        f"{display_name(cfg.model.tag)}: attention to sorting signals "
        f"(n={int(payload['n_proteins']):,} proteins)"
    )
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig5_attention")[0]


def figure_ablation(cfg) -> Path | None:
    """Figure 6: per-compartment degradation under N-terminal ablation."""
    path = cfg.paths.analysis / f"{cfg.model.tag}_ablation.csv"
    if not path.exists():
        logger.warning("Skipping ablation figure: %s missing", path)
        return None
    ablation = pd.read_csv(path)

    pivot = ablation.pivot_table(index="class", columns="ablation", values="delta")
    pivot = pivot.reindex([c for c in LOCALIZATION_CLASSES if c in pivot.index])

    # Order columns as dose-response series first, controls last, rather than the
    # alphabetical default -- which would separate each series from its control.
    mode_order = {"nterm_delete": 0, "nterm_mask": 1, "shuffle_nterm": 2, "internal_delete": 3}
    key = (
        ablation[["ablation", "mode", "prefix"]]
        .drop_duplicates()
        .set_index("ablation")
    )
    ordered = sorted(
        pivot.columns,
        key=lambda c: (mode_order.get(key.loc[c, "mode"], 99), int(key.loc[c, "prefix"])),
    )
    pivot = pivot[ordered]

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 2.6))
    limit = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([CLASS_ABBREVIATIONS.get(c, c) for c in pivot.index])
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_title(f"{display_name(cfg.model.tag)}: change in per-compartment MCC under ablation")
    annotate_heatmap(ax, pivot.to_numpy(), fmt="{:+.2f}", threshold=0.0)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=r"$\Delta$ MCC")
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig6_ablation")[0]


def figure_scale_comparison(cfg) -> Path | None:
    """Figure 7: layer profiles across model scales, on a relative-depth axis."""
    paths = sorted(cfg.paths.probes.glob("*_selectivity.csv"))
    if len(paths) < 2:
        logger.warning("Skipping scale comparison: need >=2 models, found %d", len(paths))
        return None

    frames = [pd.read_csv(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    n_blocks = {"esm2_35m": 12, "esm2_150m": 30, "esm2_650m": 33, "prott5_xl": 24}
    order = ["esm2_35m", "esm2_150m", "esm2_650m", "prott5_xl"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.4))
    for pooling, ax in zip(("mean", "nterm"), axes, strict=True):
        subset = combined[combined["pooling"] == pooling]
        for model in [m for m in order if m in set(subset["model"])]:
            group = subset[subset["model"] == model].sort_values("layer")
            blocks = n_blocks.get(model)
            if not blocks:
                continue
            # Relative depth makes encoders of different depth comparable.
            x = group["layer"] / blocks
            ax.plot(x, group["real"], marker="o", markersize=2.5, label=model)
            if "real_sd" in group:
                ax.fill_between(
                    x, group["real"] - group["real_sd"].fillna(0),
                    group["real"] + group["real_sd"].fillna(0),
                    alpha=0.2, linewidth=0,
                )
        ax.set_xlabel("Relative depth (layer / blocks)")
        ax.set_ylabel("Macro MCC")
        ax.set_title(f"{pooling} pooling")
    axes[0].legend(title="Model")
    add_panel_labels(axes)
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig7_scale_comparison")[0]


def figure_appendix_pooling(cfg) -> Path | None:
    """Appendix figure: the three pooling windows, and the MLP capacity control.

    Both panels answer capacity/locus questions that the main-text figures leave
    open: whether a C-terminal window recovers the compartments an \\nterm{} window
    misses, and whether a non-linear probe extracts information the linear probe
    cannot reach.
    """
    base = cfg.paths.probes / f"{cfg.model.tag}_selectivity.csv"
    cterm = cfg.paths.probes / f"{cfg.model.tag}_cterm_selectivity.csv"
    mlp = cfg.paths.probes / f"{cfg.model.tag}_mlp_selectivity.csv"
    if not base.exists():
        logger.warning("Skipping appendix pooling figure: %s missing", base)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.4))

    frame = pd.read_csv(base)
    for pooling in ("mean", "nterm"):
        group = frame[frame["pooling"] == pooling].sort_values("layer")
        axes[0].plot(group["layer"], group["real"], marker="o", markersize=2.5, label=pooling)
    if cterm.exists():
        group = pd.read_csv(cterm).sort_values("layer")
        axes[0].plot(group["layer"], group["real"], marker="o", markersize=2.5, label="cterm")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Macro MCC")
    axes[0].set_title("Pooling window")
    axes[0].legend()

    if mlp.exists():
        linear = frame[frame["pooling"] == "mean"].set_index("layer")["real"]
        nonlinear = pd.read_csv(mlp)
        nonlinear = nonlinear[nonlinear["pooling"] == "mean"].set_index("layer")["real"]
        axes[1].plot(linear.index, linear.to_numpy(), marker="o", markersize=2.5, label="linear")
        axes[1].plot(
            nonlinear.index, nonlinear.to_numpy(), marker="s", markersize=2.5, label="MLP"
        )
        gap = (nonlinear - linear).dropna()
        axes[1].fill_between(gap.index, linear.reindex(gap.index), nonlinear.reindex(gap.index),
                             alpha=0.2, linewidth=0)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Macro MCC")
        axes[1].set_title("Probe capacity (mean pooling)")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    add_panel_labels(axes)
    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "figA1_appendix_controls")[0]



def figure_sorting_pathways(cfg) -> Path | None:
    """Schematic of eukaryotic protein sorting routes and determinant positions.

    Unlike every other figure here this one plots no data: it is a schematic for
    the background document. It is generated rather than hand-drawn so that it
    inherits the same palette, typography and sizing as the result figures, and so
    that it is regenerated and hash-registered by the same pipeline.

    Panel (a) traces the routes from the cytosolic ribosome; panel (b) shows where
    along the sequence each compartment's determinant sits, which is the asymmetry
    the probing experiments exploit.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    # Determinant classes, coloured consistently across both panels.
    CLASS_COLOR = {
        "nterm": "#0072B2",       # cleavable N-terminal presequence
        "internal": "#009E73",    # internal, non-cleavable
        "cterm": "#D55E00",       # C-terminal motif
        "distributed": "#CC79A7",  # spread along the sequence
        "none": "#999999",        # no positive signal
    }
    CLASS_LABEL = {
        "nterm": "Cleavable N-terminal presequence",
        "internal": "Internal signal",
        "cterm": "C-terminal motif",
        "distributed": "Distributed determinant",
        "none": "No positive signal",
    }

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.32)
    ax_a, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    # ---------------------------------------------------------------- panel a --
    def box(ax, x, y, text, cls, w=1.72, h=0.42):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.0, edgecolor=CLASS_COLOR[cls],
            facecolor=CLASS_COLOR[cls] + "22"))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.0)

    def arrow(ax, xy_from, xy_to, cls, label=None, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            xy_from, xy_to, arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=CLASS_COLOR[cls],
            connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3))
        if label:
            mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
            ax.text(mx, my + 0.17, label, ha="center", va="bottom",
                    fontsize=6.0, color=CLASS_COLOR[cls])

    box(ax_a, 1.15, 3.0, "Cytosolic\nribosome", "none", w=1.5, h=0.62)

    # Non-secretory routes, each labelled with its determinant.
    direct = [
        (4.3, 5.05, "Mitochondrion", "nterm", "transit peptide"),
        (4.3, 4.25, "Plastid", "nterm", "transit peptide"),
        (4.3, 3.45, "Nucleus", "internal", "NLS"),
        (4.3, 2.65, "Peroxisome", "cterm", "PTS1"),
        (4.3, 1.85, "Cytoplasm", "none", "no signal: default"),
    ]
    for x, y, name, cls, lab in direct:
        arrow(ax_a, (1.95, 3.0), (x - 0.9, y), cls)
        box(ax_a, x, y, name, cls)
        ax_a.text(x + 0.95, y, lab, ha="left", va="center", fontsize=6.0,
                  color=CLASS_COLOR[cls])

    # The secretory pathway is a chain rather than a single hop.
    box(ax_a, 4.3, 0.85, "Endoplasmic\nreticulum", "nterm", h=0.56)
    arrow(ax_a, (1.95, 3.0), (3.4, 0.85), "nterm")
    # Set horizontally in the clear wedge below the arrow: rotated along it, the
    # label either crosses the arrow fan or runs into the ER box.
    ax_a.text(2.38, 1.18, "signal peptide", ha="center", va="center", fontsize=6.0,
              color=CLASS_COLOR["nterm"])

    box(ax_a, 7.0, 0.85, "Golgi\napparatus", "distributed", h=0.56)
    arrow(ax_a, (5.16, 0.85), (6.14, 0.85), "distributed")

    for y, name, cls in [(1.85, "Cell membrane", "distributed"),
                         (0.85, "Extracellular", "nterm"),
                         (-0.15, "Lysosome / vacuole", "distributed")]:
        # Flow along the secretory pathway is drawn neutral: the determinant that
        # selected the route acted on entry, not at this step.
        arrow(ax_a, (7.86, 0.85), (8.84, y), "none", rad=0.0 if y == 0.85 else 0.12)
        box(ax_a, 9.7, y, name, cls)

    ax_a.text(7.0, 1.30, "retained by TM-helix length", ha="center", va="bottom",
              fontsize=6.0, color=CLASS_COLOR["distributed"], style="italic")
    ax_a.text(9.7, -0.72, "M6P tag added in Golgi", ha="center", va="top",
              fontsize=6.0, color=CLASS_COLOR["distributed"], style="italic")

    ax_a.set_xlim(0.1, 11.3)
    ax_a.set_ylim(-0.75, 5.55)
    ax_a.axis("off")
    ax_a.set_title("Routes from the cytosolic ribosome", fontsize=8.5, pad=2)

    # ---------------------------------------------------------------- panel b --
    # Where along the sequence each compartment's determinant lies. Spans are
    # schematic, not to scale: PTS1 is three residues, a signal peptide 15-30.
    rows = [
        ("Extracellular",   "nterm",       [(0.00, 0.09)], "signal peptide"),
        ("Mitochondrion",   "nterm",       [(0.00, 0.11)], "transit peptide"),
        ("Plastid",         "nterm",       [(0.00, 0.13)], "transit peptide"),
        ("Nucleus",         "internal",    [(0.34, 0.42)], "NLS (position varies)"),
        ("Peroxisome",      "cterm",       [(0.975, 1.00)], "PTS1 (3 residues)"),
        ("ER (residents)",  "cterm",       [(0.96, 1.00)], "KDEL"),
        ("Cell membrane",   "distributed", [(0.12, 0.20), (0.34, 0.42),
                                            (0.56, 0.64), (0.78, 0.86)], "TM helices"),
        ("Golgi apparatus", "distributed", [], "TM-helix length, not a position"),
        ("Lysosome/vacuole", "distributed", [], "M6P tag: conformational, not linear"),
        ("Cytoplasm",       "none",        [], "no positive signal"),
    ]

    for i, (name, cls, spans, note) in enumerate(rows):
        y = len(rows) - i
        ax_b.add_patch(FancyBboxPatch(
            (0, y - 0.17), 1.0, 0.34,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            linewidth=0.6, edgecolor="0.75", facecolor="0.94"))
        for x0, x1 in spans:
            ax_b.add_patch(FancyBboxPatch(
                (x0, y - 0.17), max(x1 - x0, 0.012), 0.34,
                boxstyle="round,pad=0.002,rounding_size=0.01",
                linewidth=0, facecolor=CLASS_COLOR[cls]))
        ax_b.text(-0.02, y, name, ha="right", va="center", fontsize=7.0)
        ax_b.text(1.03, y, note, ha="left", va="center", fontsize=6.2,
                  color=CLASS_COLOR[cls])

    axis_y = 0.30
    ax_b.annotate("", xy=(1.0, axis_y), xytext=(0.0, axis_y),
                  arrowprops=dict(arrowstyle="-", linewidth=0.7, color="0.55"))
    ax_b.text(-0.03, axis_y, "N", ha="right", va="center", fontsize=7.5,
              fontweight="bold")
    ax_b.text(1.03, axis_y, "C", ha="left", va="center", fontsize=7.5,
              fontweight="bold")
    ax_b.text(0.5, axis_y - 0.30, "position along sequence", ha="center",
              va="center", fontsize=6.5, color="0.45")

    ax_b.set_xlim(-0.34, 1.78)
    ax_b.set_ylim(-0.25, len(rows) + 0.75)
    ax_b.axis("off")
    ax_b.set_title("Where the determinant lies along the sequence",
                   fontsize=8.5, pad=2)

    # Figure-level legend, below both panels, so it cannot collide with the axis.
    handles = [plt.Line2D([], [], color=CLASS_COLOR[k], linewidth=4,
                          label=CLASS_LABEL[k])
               for k in ("nterm", "internal", "cterm", "distributed", "none")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5,
               frameon=False, handlelength=1.5, bbox_to_anchor=(0.5, -0.005))

    add_panel_labels([ax_a, ax_b], dx=-6.0, dy=2.0)
    fig.subplots_adjust(bottom=0.12)
    return save_figure(fig, cfg.paths.figures / "figS1_sorting_pathways")[0]



def figure_compartment_difficulty(cfg) -> Path | None:
    """A priori tractability of each compartment from sequence alone.

    Like the sorting schematic this plots no measurements. It sets out what the
    cell biology implies about how legible each compartment is to a sequence
    model, so that the prediction can be stated before any model is fitted and
    checked afterwards. Deliberately kept free of results, so the background
    document stays independent of the companion study.
    """
    from matplotlib.patches import FancyBboxPatch

    FAVOURABLE, PARTIAL, ADVERSE, NA = "#009E73", "#E69F00", "#D55E00", "#BBBBBB"

    # Dashes are written as literal en/em dashes: this is matplotlib, not LaTeX,
    # so "--" would be typeset as two hyphens.

    # (compartment, tier, [(cell text, verdict) per property])
    # Properties: discrete signal? / positionally fixed? / linear in sequence? / length
    rows = [
        ("Extracellular",    0, [("yes", FAVOURABLE), ("N-term", FAVOURABLE),
                                 ("yes", FAVOURABLE), ("15\u201330 aa", FAVOURABLE)]),
        ("Mitochondrion",    0, [("subset only", PARTIAL), ("N-term", FAVOURABLE),
                                 ("yes", FAVOURABLE), ("20\u201360 aa", FAVOURABLE)]),
        ("Plastid",          0, [("yes", FAVOURABLE), ("N-term", FAVOURABLE),
                                 ("yes", FAVOURABLE), ("30\u201380 aa", FAVOURABLE)]),
        ("Nucleus",          1, [("yes", FAVOURABLE), ("internal", PARTIAL),
                                 ("yes", FAVOURABLE), ("7\u201320 aa", FAVOURABLE)]),
        ("ER (residents)",   1, [("yes", FAVOURABLE), ("C-term", PARTIAL),
                                 ("yes", FAVOURABLE), ("4 aa", PARTIAL)]),
        ("Peroxisome",       1, [("yes", FAVOURABLE), ("C-term", PARTIAL),
                                 ("yes", FAVOURABLE), ("3 aa", ADVERSE)]),
        ("Cell membrane",    2, [("partly", PARTIAL), ("distributed", ADVERSE),
                                 ("yes", FAVOURABLE), ("20 aa $\\times$ n", PARTIAL)]),
        ("Golgi apparatus",  2, [("no", ADVERSE), ("\u2014", NA),
                                 ("no: helix length", ADVERSE), ("\u2014", NA)]),
        ("Lysosome/vacuole", 2, [("no", ADVERSE), ("\u2014", NA),
                                 ("no: M6P tag", ADVERSE), ("\u2014", NA)]),
        ("Cytoplasm",        2, [("no: default", ADVERSE), ("\u2014", NA),
                                 ("no: absence", ADVERSE), ("\u2014", NA)]),
    ]

    columns = ["Discrete\nsignal?", "Positionally\nfixed?",
               "Linear in\nsequence?", "Signal\nlength"]
    tiers = [
        ("Cleavable N-terminal presequence:\nmost tractable from a short window", "#0072B2"),
        ("Signal exists but is internal or C-terminal:\nneeds the whole sequence", "#56B4E9"),
        ("No linear determinant: physical,\npost-translational or negative", "#999999"),
    ]

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 4.35))

    col_x = [0.0, 1.30, 2.60, 3.95]
    col_w = [1.18, 1.18, 1.28, 1.05]

    for j, (name, x, w) in enumerate(zip(columns, col_x, col_w, strict=True)):
        ax.text(x + w / 2, len(rows) + 0.62, name, ha="center", va="bottom",
                fontsize=6.8, linespacing=1.25)

    for i, (compartment, tier, cells) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(-0.16, y, compartment, ha="right", va="center", fontsize=7.2)
        for (text, colour), x, w in zip(cells, col_x, col_w, strict=True):
            ax.add_patch(FancyBboxPatch(
                (x, y - 0.34), w, 0.68,
                boxstyle="round,pad=0.004,rounding_size=0.05",
                linewidth=0.7, edgecolor=colour, facecolor=colour + "26"))
            ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=6.4,
                    color="black" if colour != NA else "0.45")

    # Tier brackets down the right-hand side.
    bracket_x = col_x[-1] + col_w[-1] + 0.22
    for tier, (label, colour) in enumerate(tiers):
        members = [i for i, r in enumerate(rows) if r[1] == tier]
        top = len(rows) - 1 - min(members) + 0.40
        bottom = len(rows) - 1 - max(members) - 0.40
        ax.plot([bracket_x, bracket_x], [bottom, top], color=colour, linewidth=2.4,
                solid_capstyle="butt")
        ax.text(bracket_x + 0.14, (top + bottom) / 2, label, ha="left",
                va="center", fontsize=6.5, color=colour, linespacing=1.3)

    ax.text(-0.16, len(rows) + 0.62, "Compartment", ha="right", va="bottom",
            fontsize=6.8, style="italic", color="0.35")

    # Difficulty arrow on the far left.
    ax.annotate("", xy=(-1.62, -0.45), xytext=(-1.62, len(rows) - 0.55),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="0.45"))
    ax.text(-1.74, (len(rows) - 1) / 2, "expected difficulty from sequence alone",
            rotation=90, ha="center", va="center", fontsize=6.6, color="0.45")

    handles = [plt.Line2D([], [], color=c, linewidth=4, label=l) for l, c in
               [("favourable", FAVOURABLE), ("partial", PARTIAL),
                ("unfavourable", ADVERSE), ("not applicable", NA)]]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, -0.13),
              ncol=4, fontsize=6.5, frameon=False, handlelength=1.5)

    ax.set_xlim(-1.95, bracket_x + 2.30)
    ax.set_ylim(-1.05, len(rows) + 1.15)
    ax.axis("off")
    return save_figure(fig, cfg.paths.figures / "figS2_compartment_difficulty")[0]



def figure_measured_difficulty(cfg) -> Path | None:
    """Measured per-compartment MCC, grouped by the sorting regime of each compartment.

    The empirical counterpart to the schematic in the background document: it asks
    whether the regimes the sorting biology licenses line up with which
    compartments a probe decodes well. Class size is checked as the obvious rival
    explanation, since the compartments differ in prevalence by a factor of ~27.

    This is a consistency check, not a prediction. The regimes are read off the
    sorting literature rather than fitted, but they were assigned by authors who
    had already seen these measurements, so the correlation below cannot be read
    as an out-of-sample test.
    """
    path = cfg.paths.probes / f"{cfg.model.tag}_per_class.csv"
    if not path.exists():
        logger.warning("Skipping measured-difficulty figure: %s missing", path)
        return None

    from scipy.stats import spearmanr

    frame = pd.read_csv(path)
    frame = frame[(frame["layer"] == frame["layer"].max()) & (frame["pooling"] == "mean")]
    stats = frame.groupby("class")["mcc"].agg(["mean", "std", "size"])
    counts = frame[frame["seed"] == frame["seed"].min()].set_index("class")["n_positive"]
    stats["n"] = counts
    stats["tier"] = stats.index.map(REGIME)
    stats = stats.sort_values("mean")

    rho, pval = spearmanr(stats["tier"], stats["mean"])
    rho_n, pval_n = spearmanr(stats["n"], stats["mean"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 3.0))
    y = np.arange(len(stats))
    ax.barh(y, stats["mean"], xerr=stats["std"].fillna(0),
            color=[REGIME_COLOURS[t] for t in stats["tier"]],
            edgecolor=REGIME_EDGE, linewidth=0.4, height=0.66,
            error_kw=dict(elinewidth=0.9, capsize=2, ecolor="0.25"))

    for i, (name, row) in enumerate(stats.iterrows()):
        ax.text(row["mean"] + row["std"] + 0.015, i, f"n={int(row['n']):,}",
                va="center", ha="left", fontsize=6.0, color="0.4")

    ax.set_yticks(y)
    ax.set_yticklabels([CLASS_ABBREVIATIONS.get(c, c) for c in stats.index])
    ax.set_xlabel("Test-set MCC (mean of five probe fits, error bars $\\pm$1 s.d.)")
    ax.set_xlim(0, 1.0)
    ax.set_title(
        f"{display_name(cfg.model.tag)}: measured decodability by sorting regime",
        fontsize=8.5)

    handles = [plt.Line2D([], [], color=REGIME_COLOURS[t], linewidth=5,
                          label=REGIME_LABELS[t]) for t in (1, 2, 3)]
    ax.legend(handles=handles, loc="lower right", fontsize=6.3, frameon=False,
              handlelength=1.4)

    # State the test and the rival explanation on the figure itself.
    ax.text(0.985, 0.30,
            f"Spearman(regime, MCC) = {rho:.2f}, $p$ = {pval:.3f}\n"
            f"Spearman(class size, MCC) = {rho_n:.2f}, $p$ = {pval_n:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
            color="0.3", linespacing=1.4)

    fig.tight_layout()
    return save_figure(fig, cfg.paths.figures / "fig8_measured_difficulty")[0]



#: Composition feature sets, in increasing richness, with their dimensionality.
#: Order is the plotting order, so the incremental value of each block reads
#: down the panel.
_COMPOSITION_SETS = (
    ("aac", "Amino-acid\ncomposition", 20),
    ("dipeptide", "Dipeptide\ncomposition", 400),
    ("aac_physchem", "Composition +\nphysico-chemical", 26),
    ("all", "All of\nthe above", 426),
)


def figure_composition_baseline(cfg) -> Path | None:
    """Figure 9: how much of the signal explicit sequence composition carries.

    Panel (a) places the composition feature sets against two reference points
    from the same probing protocol: the encoder's input embedding and its best
    layer. Panel (b) is the informative one -- a paired dot plot of composition
    against the best layer per compartment, which shows where a learned
    representation adds something beyond bag-of-residues statistics and where it
    adds almost nothing.
    """
    baseline = cfg.paths.probes / "composition_baseline.csv"
    per_class = cfg.paths.probes / "composition_baseline_per_class.csv"
    plm_path = cfg.paths.probes / f"{cfg.model.tag}_per_class.csv"
    probe_path = cfg.paths.probes / f"{cfg.model.tag}_probe_results.csv"
    missing = [p for p in (baseline, per_class, plm_path, probe_path) if not p.exists()]
    if missing:
        logger.warning("Skipping composition figure: %s missing", missing[0])
        return None

    frame = pd.read_csv(baseline)
    real = frame[frame["task"] == "real"].groupby("pooling")["macro_mcc"].agg(["mean", "std"])

    # Reference points: the same probe on the encoder's weakest and best layers.
    profile = pd.read_csv(probe_path)
    profile = profile[(profile["pooling"] == "mean") & (profile["task"] == "real")]
    profile = profile.groupby("layer")["macro_mcc"].mean()
    layer0, best_layer = float(profile.loc[0]), int(profile.idxmax())
    layer_best = float(profile.max())

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN, 3.35), gridspec_kw={"width_ratios": [1.0, 1.3]}
    )

    # --- (a) feature sets against the encoder ---------------------------------
    ax = axes[0]
    keys = [k for k, _, _ in _COMPOSITION_SETS if k in real.index]
    y = np.arange(len(keys))
    ax.barh(y, [real.loc[k, "mean"] for k in keys],
            xerr=[real.loc[k, "std"] for k in keys],
            color=OKABE_ITO[0], edgecolor="none", height=0.6,
            error_kw=dict(elinewidth=0.9, capsize=2, ecolor="0.25"))
    for i, k in enumerate(keys):
        ax.text(real.loc[k, "mean"] + real.loc[k, "std"] + 0.012, i,
                f"{real.loc[k, 'mean']:.3f}", va="center", ha="left",
                fontsize=6.2, color="0.35")

    for value, label, style in (
        (layer0, "input embedding", (0, (1, 1.6))),
        (layer_best, f"best layer ({best_layer})", (0, (4, 1.8))),
    ):
        ax.axvline(value, color="0.45", linewidth=0.8, linestyle=style, zorder=0)
        ax.text(value + 0.012, 0.99, f"{label}\n{value:.3f}",
                transform=ax.get_xaxis_transform(), fontsize=6.0, color="0.35",
                va="top", ha="left", linespacing=1.3)

    ax.set_yticks(y)
    ax.set_yticklabels([label for k, label, _ in _COMPOSITION_SETS if k in real.index],
                       fontsize=6.4, linespacing=1.25)
    ax.set_xlabel("Macro MCC")
    ax.set_xlim(0, 0.72)
    ax.set_ylim(-1.15, len(keys) - 0.35)
    ax.invert_yaxis()
    ax.set_title("No language model", fontsize=8.0)

    # --- (b) per compartment, composition against the best layer --------------
    ax = axes[1]
    comp = pd.read_csv(per_class)
    comp = comp[comp["pooling"] == "all"].groupby("class")["mcc"].mean()
    plm_frame = pd.read_csv(plm_path)
    plm_frame = plm_frame[
        (plm_frame["layer"] == plm_frame["layer"].max()) & (plm_frame["pooling"] == "mean")
    ]
    plm = plm_frame.groupby("class")["mcc"].mean().reindex(comp.index)
    recovered = (comp / plm).sort_values()

    order = list(recovered.index)
    y = np.arange(len(order))
    for i, name in enumerate(order):
        colour = REGIME_COLOURS[REGIME[name]]
        ax.plot([comp[name], plm[name]], [i, i], color=colour, linewidth=2.4,
                solid_capstyle="round", zorder=1)
        # Open marker = composition only, filled = the learned representation.
        ax.plot(comp[name], i, "o", markersize=4.2, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=1.1, zorder=2)
        ax.plot(plm[name], i, "o", markersize=4.2, color=colour,
                markeredgecolor=REGIME_EDGE, markeredgewidth=0.4, zorder=2)
        ax.text(plm[name] + 0.022, i, f"{recovered[name] * 100:.0f}%",
                va="center", ha="left", fontsize=6.0, color="0.35")

    ax.set_yticks(y)
    ax.set_yticklabels([CLASS_ABBREVIATIONS.get(c, c) for c in order])
    ax.set_xlabel("Test-set MCC")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, len(order) - 0.5)
    ax.set_title("Composition against the best layer, per compartment", fontsize=8.0)

    marker_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=4.2,
                   markerfacecolor="white", markeredgecolor="0.35",
                   markeredgewidth=1.1, label="Composition only"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=4.2,
                   color="0.35", label=f"{display_name(cfg.model.tag)}, layer {best_layer}"),
    ]
    ax.legend(handles=marker_handles, loc="lower right", fontsize=6.2,
              frameon=False, handletextpad=0.4, borderpad=0.2,
              bbox_to_anchor=(1.0, -0.02))

    add_panel_labels(axes)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(
        handles=[plt.Line2D([], [], color=REGIME_COLOURS[r], linewidth=3,
                            label=REGIME_LABELS[r]) for r in (1, 2, 3)],
        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.2,
        frameon=False, handlelength=1.3, handletextpad=0.5, columnspacing=1.8,
    )
    return save_figure(fig, cfg.paths.figures / "fig9_composition_baseline")[0]


FIGURES = {
    "dataset": figure_dataset_overview,
    "layers": figure_layer_profile,
    "per_class": figure_per_class_layers,
    "cka": figure_cka,
    "attention": figure_attention,
    "ablation": figure_ablation,
    "scales": figure_scale_comparison,
    "appendix": figure_appendix_pooling,
    "sorting": figure_sorting_pathways,
    "difficulty": figure_compartment_difficulty,
    "measured_difficulty": figure_measured_difficulty,
    "composition": figure_composition_baseline,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--only", nargs="*", choices=sorted(FIGURES), default=None,
        help="Render only these figures (default: all available)",
    )
    args = parser.parse_args()
    cfg = resolve(args, "make_figures")
    set_style()

    selected = args.only or list(FIGURES)
    written = []
    for name in selected:
        result = FIGURES[name](cfg)
        if result is not None:
            written.append(result)
            logger.info("Wrote %s", result)

    logger.info("Rendered %d/%d figures into %s", len(written), len(selected), cfg.paths.figures)


if __name__ == "__main__":
    main()
