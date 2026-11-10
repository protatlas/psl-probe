# Probing Protein Language Model Representations for Subcellular
Localization: Layer Depth, Sequence Locus, and the Role of Sorting Signals

Layer-wise probing of protein language models (pLMs) for subcellular
localization signals.

## Research question

Protein language models such as ESM-2 and ProtT5 encode strong predictors of a
protein's subcellular localization in their frozen representations, but it is
unclear *where* that information lives: concentrated in N-terminal sorting
signals (signal peptides, mitochondrial/chloroplast transit peptides) the way a
biologist would expect, or spread diffusely across the sequence as a
composition-like feature. This project probes that question layer by layer,
model by model, with:

- **Linear probes** fit on frozen per-layer embeddings for a ten-compartment
  multi-label localization taxonomy (the [labels.py](code/psl_probe/data/labels.py)
  taxonomy follows DeepLoc 2.0, Thumuluri et al., *NAR* 2022), with an MLP
  capacity control and selectivity-style control tasks (Hewitt & Liang, *EMNLP*
  2019).
- **Homology-aware splits** (MMseqs2 clustering at 30% identity) so probe
  accuracy can't be inflated by near-duplicate sequences crossing the
  train/test boundary — see [splits.py](code/psl_probe/data/splits.py).
- **Sequence ablation** (N-terminal deletion/masking/shuffling vs. a
  length-matched internal-deletion control) to test whether decodability is
  causally tied to the N-terminus.
- **Pooling contrasts** (whole-protein mean vs. N-terminal-window pooling) at
  a fixed layer.
- **CKA** between layers, to see whether the layer where localization becomes
  linearly decodable coincides with a real change in representational
  geometry.
- **Attention-to-signal alignment**, scored against a position-matched null so
  a generic N-terminal attention bias isn't mistaken for signal recognition.
- **A composition-only baseline** (hydrophobicity, length, cysteine content,
  ...) with no language model at all, to bound how much of the "distributed"
  signal is just amino-acid composition.

Models covered: ESM-2 at three scales (35M / 150M / 650M params) and ProtT5-XL.

## Repository layout

```
configs/                 Model and experiment YAML configs
  base.yaml              Shared defaults (paths, data filters, probe/analysis params)
  esm2_35m.yaml           ESM-2 t12 35M  — smoke-test scale
  esm2_150m.yaml           ESM-2 t30 150M — scale-comparison point
  esm2_650m.yaml           ESM-2 t33 650M — primary probing target
  prott5.yaml              ProtT5-XL-U50 encoder
  experiments/             Named, trackable experiment definitions (see below)

code/
  run_experiment.py       Single tracked entry point for every reported result
  Makefile                Convenience targets wrapping the CLIs directly
  psl_probe/
    config.py              Typed dataclass config, hydrated from YAML + --set overrides
    tracking.py            Run-level experiment tracking (local/wandb/mlflow)
    data/                  UniProt acquisition, label taxonomy, homology-aware splits
    embeddings/            Model wrapper (ESM-2/ProtT5) and pooling strategies
    probes/                Linear/MLP probes, trainer, control-task labels
    analysis/              CKA, attention alignment, ablation, composition baseline, metrics, plotting
    cli/                   One module per pipeline stage (see Pipeline below)
  tests/                   Unit tests for the above
```

Results, datasets, figures, and logs are written outside of version control to
paths configured in `configs/base.yaml` (`datasets/`, `results/`,
`visualization/figures/`, `logs/`) and are not part of this repo.

## Pipeline

Each stage is both a standalone CLI (`python -m psl_probe.cli.<stage>
--config <model-config.yaml>`) and a named stage usable from
`run_experiment.py`, which additionally records the exact command, code
version, environment, seeds, config and output hashes for every run that
feeds the manuscript:

```bash
python code/run_experiment.py --config configs/experiments/main_650m.yaml
```

| Stage | Module | Purpose |
|---|---|---|
| 1 | `prepare_data` | Build the homology-partitioned Swiss-Prot benchmark |
| 2 | `extract_embeddings` | Extract per-layer pLM representations |
| 3 | `run_probes` | Fit probes across layer × pooling × control-task |
| 3b | `run_composition_baseline` | Fit probes to explicit sequence composition (no pLM) |
| 4 | `run_analysis` | CKA, attention alignment, bootstrap confidence intervals |
| 4b | `run_ablation` | N-terminal ablation sweep with multi-seed intervals |
| 4c | `run_pooling_contrast` | Mean- vs. N-terminal-pooling contrast at a fixed layer |
| — | `make_split_variants` | Alternate-homology-threshold splits (leakage sensitivity, Appendix) |
| 5 | `make_figures` | Render every manuscript figure |
| 6 | `make_tables` | Render every manuscript table as LaTeX |

Predefined experiments live in `configs/experiments/`:

- `smoke.yaml` — ESM-2 35M, capped benchmark, full pipeline in minutes. Run
  this first on a fresh checkout.
- `main_650m.yaml` — the primary ESM-2 650M results (manuscript §3.2–3.6).
- `scales.yaml` — scale comparison across ESM-2 35M/150M/650M (§3.7).
- `ablation_650m.yaml` — N-terminal ablation sweep with controls (§3.5).
- `composition_baseline.yaml` — no-pLM composition baseline (Discussion).
- `homology_sensitivity.yaml` — split-leakage sensitivity (Appendix B.3).
- `regenerate_paper_assets.yaml` — rebuild figures/tables from existing
  results only, no re-computation.

The `Makefile` wraps the same CLIs with convenience targets
(`make smoke`, `make scales`, `make ablation`, `make pooling`, `make paper`,
`make test`); run `make help` from `code/` for the full list.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
pip install -e code/
```

Requires Python ≥3.10, PyTorch ≥2.4, and a CUDA GPU for embedding extraction at
any scale beyond the 35M smoke config (24 GB is sufficient for ESM-2 650M at
fp16). MMseqs2 must be on `PATH` for homology-aware splitting.

## Quick start

```bash
cd code
make smoke     # ~8 min on an RTX 4090: full pipeline, ESM-2 35M, 5k proteins
make test      # unit tests
```

Then, for the full manuscript pipeline:

```bash
python run_experiment.py --config ../configs/experiments/main_650m.yaml
```

## Reproducibility

`run_experiment.py --deterministic` (on by default) enables deterministic
kernels. Every tracked run writes its resolved config, console log, and
artifact manifest under `logs/<run_id>/`.
