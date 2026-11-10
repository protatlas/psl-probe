"""Stage 6: render manuscript tables as LaTeX from the stored result files.

Each table is written to ``latex/tables/<name>.tex`` as a standalone ``table``
environment that ``main.tex`` pulls in with ``\\input``. When the underlying
results do not exist yet, a clearly marked placeholder carrying a ``\\TODO`` is
emitted instead, so the manuscript always compiles and never silently displays a
table with invented numbers.

Example
-------
::

    python -m psl_probe.cli.make_tables --config configs/esm2_650m.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from psl_probe.cli import add_common_arguments, resolve
from psl_probe.config import REPO_ROOT
from psl_probe.data.labels import CLASS_ABBREVIATIONS, LOCALIZATION_CLASSES
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)

#: Set from the resolved config at run time (see main); the module-level value
#: is only a fallback for direct imports.
TABLE_DIR = REPO_ROOT / "latex" / "tables"


def _escape(text: object) -> str:
    """Escape the LaTeX special characters that occur in our column values."""
    out = str(text)
    for char, replacement in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
    ):
        out = out.replace(char, replacement)
    return out


def _table(
    frame: pd.DataFrame, caption: str, label: str, column_format: str | None = None
) -> str:
    """Render a DataFrame as a booktabs ``table`` environment."""
    columns = list(frame.columns)
    fmt = column_format or ("l" + "r" * (len(columns) - 1))
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{fmt}}}",
        r"    \toprule",
        "    " + " & ".join(_escape(c) for c in columns) + r" \\",
        r"    \midrule",
    ]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.3f}" if abs(value) < 1000 else f"{value:.3g}")
            else:
                cells.append(_escape(value))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _placeholder(caption: str, label: str, source: Path) -> str:
    """A compilable stand-in that names the file it is waiting for."""
    relative = source.relative_to(REPO_ROOT) if source.is_absolute() else source
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            rf"  \caption{{{caption}}}",
            rf"  \label{{{label}}}",
            r"  \begin{tabular}{l}",
            r"    \toprule",
            r"    \textit{Not yet generated} \\",
            r"    \midrule",
            rf"    \TODO{{run the pipeline to produce \texttt{{{_escape(relative)}}},}} \\",
            r"    \TODO{then re-run \texttt{python -m psl\_probe.cli.make\_tables}} \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _emit(name: str, caption: str, label: str, source: Path,
          builder, column_format: str | None = None) -> bool:
    """Write ``name.tex``: the real table when ``source`` exists, else a placeholder."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    target = TABLE_DIR / f"{name}.tex"
    if source.exists():
        frame = builder()
        target.write_text(_table(frame, caption, label, column_format), encoding="utf-8")
        logger.info("Wrote %s from %s", target.name, source.name)
        return True
    target.write_text(_placeholder(caption, label, source), encoding="utf-8")
    logger.warning("Placeholder for %s (missing %s)", target.name, source)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    args = parser.parse_args()
    cfg = resolve(args, "make_tables")

    global TABLE_DIR
    TABLE_DIR = Path(cfg.paths.tables)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Writing tables to %s", TABLE_DIR)

    processed = cfg.paths.processed
    probes = cfg.paths.probes
    built = 0

    built += _emit(
        "tab_filter_cascade",
        "Construction of the probing benchmark. Each row reports how many "
        "Swiss-Prot entries were removed by that inclusion criterion, applied in "
        "the order shown.",
        "tab:filter_cascade",
        processed / "filter_report.csv",
        lambda: pd.read_csv(processed / "filter_report.csv").rename(
            columns={"reason": "Criterion", "n_removed": "Entries removed"}
        ),
    )

    built += _emit(
        "tab_class_summary",
        "Per-compartment composition of the benchmark. "
        "\\textit{Multi-label} counts proteins that carry the compartment together "
        "with at least one other.",
        "tab:class_summary",
        processed / "class_summary.csv",
        lambda: pd.read_csv(processed / "class_summary.csv").assign(
            Abbrev=lambda d: d["class"].map(CLASS_ABBREVIATIONS)
        ).rename(
            columns={"class": "Compartment", "n_proteins": "Proteins",
                     "prevalence": "Prevalence", "n_multi_label": "Multi-label",
                     "frac_multi_label": "Frac. multi-label"}
        )[["Compartment", "Abbrev", "Proteins", "Prevalence",
           "Multi-label", "Frac. multi-label"]],
    )

    built += _emit(
        "tab_signal_features",
        "Curated UniProt sorting-signal features per compartment. These counts "
        "bound the sample size available to the attention-alignment analysis.",
        "tab:signal_features",
        processed / "signal_summary.csv",
        lambda: pd.read_csv(processed / "signal_summary.csv").rename(
            columns={"class": "Compartment", "n_proteins": "Proteins",
                     "signal_peptide": "SIGNAL", "transit_peptide": "TRANSIT",
                     "transmembrane": "TRANSMEM", "propeptide": "PROPEP",
                     "lipid_anchor": "LIPID"}
        ),
    )

    built += _emit(
        "tab_split_summary",
        "Homology-partitioned folds. Fold 4 is held out as the test partition. "
        "Comparable per-class counts confirm that cluster-level assignment did not "
        "distort the class distribution.",
        "tab:split_summary",
        processed / "split_summary.csv",
        lambda: pd.read_csv(processed / "split_summary.csv").rename(
            columns={"fold": "Fold", "n_proteins": "Proteins", "n_clusters": "Clusters",
                     **CLASS_ABBREVIATIONS}
        ),
    )

    built += _emit(
        "tab_model_comparison",
        "Probe performance at each model's best layer, selected on the validation "
        "split. Selectivity is the real-task score minus the mean control-task score.",
        "tab:model_comparison",
        probes / f"{cfg.model.tag}_selectivity.csv",
        lambda: _best_layer_table(probes),
    )

    built += _emit(
        "tab_unmapped_terms",
        "Subcellular-location terms that matched no taxonomy rule, with their "
        "frequency. These annotations contribute no label and their proteins are "
        "dropped unless another term maps.",
        "tab:unmapped_terms",
        processed / "unmapped_terms.csv",
        lambda: pd.read_csv(processed / "unmapped_terms.csv").head(20),
    )

    built += _emit(
        "tab_composition_baseline",
        "Probes fitted to explicit sequence composition, with no language model. "
        "Protocol, splits, class weighting and threshold selection are identical "
        "to the pLM probes, so the scores sit on the same scale. "
        "\\textit{Features} is the dimensionality of the feature set; "
        "\\textit{Control} is the matched cluster-level control task.",
        "tab:composition_baseline",
        probes / "composition_baseline.csv",
        lambda: _composition_table(probes),
    )

    built += _emit(
        "tab_composition_per_class",
        "Per-compartment comparison of the composition-only probe against the "
        "best \\esmtwo{}-650M layer. \\textit{Recovered} is the ratio of the two, "
        "i.e.\\ the share of each compartment's decodable signal that explicit "
        "composition already captures. Regimes are those of "
        "\\cref{sec:methods:signal}.",
        "tab:composition_per_class",
        probes / "composition_baseline_per_class.csv",
        lambda: _composition_per_class_table(probes),
    )

    logger.info("Rendered %d/8 tables with real data into %s", built, TABLE_DIR)


#: Depth of each probed encoder, for the relative-depth column.
_MODEL_LAYERS = {"esm2_35m": 12, "esm2_150m": 30, "esm2_650m": 33, "prott5_xl": 24}

#: Display names, in ascending order of scale (dict order drives table row order).
_MODEL_NAMES = {
    "esm2_35m": "ESM-2 35M",
    "esm2_150m": "ESM-2 150M",
    "esm2_650m": "ESM-2 650M",
    "prott5_xl": "ProtT5-XL",
}


#: Display names and dimensionality of the composition feature sets, in the
#: order they are reported: increasing richness, so the incremental value of
#: each block is readable down the column.
_COMPOSITION_SETS = {
    "aac": ("Amino-acid composition", 20),
    "dipeptide": ("Dipeptide composition", 400),
    "aac_physchem": ("Composition + physico-chemical", 26),
    "all": ("All of the above", 426),
}

#: Sorting regime per compartment, mirroring Section 2.4 of the manuscript.
_REGIME = {
    "Extracellular": 1, "Mitochondrion": 1, "Plastid": 1,
    "Nucleus": 2, "Endoplasmic reticulum": 2, "Peroxisome": 2,
    "Cell membrane": 3, "Golgi apparatus": 3, "Lysosome/Vacuole": 3,
    "Cytoplasm": 3,
}


def _composition_table(probes: Path) -> pd.DataFrame:
    """Aggregate macro MCC per composition feature set, with its control task."""
    frame = pd.read_csv(probes / "composition_baseline.csv")
    real = frame[frame["task"] == "real"].groupby("pooling")["macro_mcc"].agg(["mean", "std"])
    control = frame[frame["task"] == "control"].groupby("pooling")["macro_mcc"].mean()
    rows = []
    for key, (label, width) in _COMPOSITION_SETS.items():
        if key not in real.index:
            continue
        rows.append({
            "Feature set": label,
            "Features": width,
            "Macro MCC": round(float(real.loc[key, "mean"]), 3),
            "s.d.": round(float(real.loc[key, "std"]), 3),
            "Control": round(float(control.get(key, float("nan"))), 3),
            "Selectivity": round(float(real.loc[key, "mean"] - control.get(key, 0.0)), 3),
        })
    return pd.DataFrame(rows)


def _composition_per_class_table(probes: Path) -> pd.DataFrame:
    """Composition-only MCC against the best pLM layer, per compartment.

    Sorted by the recovered fraction, which is the quantity of interest: it says
    where a frozen representation adds something beyond composition and where it
    does not.
    """
    comp = pd.read_csv(probes / "composition_baseline_per_class.csv")
    comp = comp[comp["pooling"] == "all"].groupby("class")["mcc"].mean()
    plm_path = probes / "esm2_650m_per_class.csv"
    plm_frame = pd.read_csv(plm_path)
    plm_frame = plm_frame[
        (plm_frame["layer"] == plm_frame["layer"].max()) & (plm_frame["pooling"] == "mean")
    ]
    plm = plm_frame.groupby("class")["mcc"].mean()
    table = pd.DataFrame({
        "Compartment": comp.index,
        "Regime": [_REGIME.get(c, "") for c in comp.index],
        "Composition": comp.round(3).to_numpy(),
        "ESM-2 650M": plm.reindex(comp.index).round(3).to_numpy(),
        "Recovered": (comp / plm.reindex(comp.index)).round(3).to_numpy(),
    })
    return table.sort_values("Recovered", ascending=False).reset_index(drop=True)


def _best_layer_table(probes: Path) -> pd.DataFrame:
    """Best layer per (model, pooling), with between-seed spread and relative depth.

    Relative depth (best layer / total layers) is what makes the scales
    comparable: an absolute layer index means something different in a 12-layer
    and a 33-layer encoder.
    """
    # Read the canonical file per model by name. A glob over *_selectivity.csv
    # also matches the variant sweeps -- random/id50/id90 splits, the cterm and
    # mlp pools, the learning-rate grid -- which all carry the same `model` tag.
    # The groupby idxmax below then silently selects the best row across those
    # variants, which put the inflated random-split score (0.610) into this table
    # as if it were the model's performance, contradicting the main text (0.558)
    # and displaying precisely the artefact the paper warns against.
    paths = [probes / f"{tag}_selectivity.csv" for tag in _MODEL_NAMES]
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"no canonical selectivity files among {[p.name for p in paths]}")
    combined = pd.concat(frames, ignore_index=True)
    best = combined.loc[combined.groupby(["model", "pooling"])["real"].idxmax()].copy()

    # Cell values are LaTeX-escaped downstream, so keep every column plain text
    # and give the standard deviation its own column rather than embedding math.
    best["Rel. depth"] = [
        round(row.layer / _MODEL_LAYERS[row.model], 2)
        if row.model in _MODEL_LAYERS else float("nan")
        for row in best.itertuples()
    ]
    best["Model"] = best["model"].map(_MODEL_NAMES).fillna(best["model"])
    best["_order"] = best["model"].map(
        {tag: i for i, tag in enumerate(_MODEL_LAYERS)}
    ).fillna(99)

    out = best.rename(
        columns={"pooling": "Pooling", "layer": "Best layer", "real": "Macro MCC",
                 "real_sd": "s.d.", "control": "Control", "selectivity": "Selectivity"}
    )
    columns = ["Model", "Pooling", "Best layer", "Rel. depth", "Macro MCC"]
    if "s.d." in out:
        columns.append("s.d.")
    columns += ["Control", "Selectivity"]
    return (
        out.sort_values(["_order", "Pooling"])[columns].reset_index(drop=True)
    )


if __name__ == "__main__":
    main()
