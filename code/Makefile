# Pipeline driver. Every target is idempotent: stages skip work whose output
# already exists, so `make all` can be re-run safely after an interruption.
#
#   make all          full pipeline for CONFIG (default: ESM-2 650M)
#   make smoke        fast end-to-end check on 5k entries with ESM-2 35M
#   make scales       layer sweep across all three ESM-2 scales
#   make ablation     sequence-ablation sweep with multi-seed intervals
#   make pooling      N-terminal vs whole-protein pooling contrast
#   make test         unit tests
#   make paper        build the manuscript

PY      := python3
CONFIGS ?= ../configs
CONFIG  ?= $(CONFIGS)/esm2_650m.yaml
LAYER   ?= 24
SEEDS   ?= 5
PREFIX  ?= 70
THREADS ?= 8
# Extra flags appended to every stage of a sweep, e.g.
#   make scales EXTRA="--set paths.embeddings=results/embeddings_v2"
EXTRA   ?=

.PHONY: all data embeddings probes analysis ablation pooling figures tables paper \
        smoke scales test lint clean-results help

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

all: data embeddings probes analysis figures tables

# -- stage 1: benchmark ------------------------------------------------------
data:
	$(PY) -m psl_probe.cli.prepare_data --config $(CONFIG) --threads $(THREADS)

# -- stage 2: representations ------------------------------------------------
embeddings:
	$(PY) -m psl_probe.cli.extract_embeddings --config $(CONFIG)

# -- stage 3: probes ---------------------------------------------------------
probes:
	$(PY) -m psl_probe.cli.run_probes --config $(CONFIG)

# -- stage 4: analyses -------------------------------------------------------
analysis:
	$(PY) -m psl_probe.cli.run_analysis --config $(CONFIG)

# Ablation re-embeds the benchmark once per (mode, prefix); run it explicitly.
# SEEDS>1 refits every probe so intervals include probe-fitting variance.
ablation:
	$(PY) -m psl_probe.cli.run_ablation --config $(CONFIG) --layer $(LAYER) \
		--seeds $(SEEDS) --contrast internal_delete:$(PREFIX)

# Contrast N-terminal against whole-protein pooling at the selected layer.
pooling:
	$(PY) -m psl_probe.cli.run_pooling_contrast --config $(CONFIG) --layer $(LAYER) \
		--reference mean --comparison nterm --seeds $(SEEDS)

# -- stage 5/6: manuscript assets --------------------------------------------
figures:
	$(PY) -m psl_probe.cli.make_figures --config $(CONFIG)

tables:
	$(PY) -m psl_probe.cli.make_tables --config $(CONFIG)

paper: figures tables
	$(MAKE) -C ../latex

# -- convenience -------------------------------------------------------------
# Fast path: small model, capped parse, relaxed homology threshold, few epochs.
# Exercises every stage in minutes rather than hours.
smoke:
	$(PY) -m psl_probe.cli.prepare_data --config $(CONFIGS)/esm2_35m.yaml \
		--limit 5000 --threads $(THREADS) --set data.homology_identity=0.5
	$(PY) -m psl_probe.cli.extract_embeddings --config $(CONFIGS)/esm2_35m.yaml
	$(PY) -m psl_probe.cli.run_probes --config $(CONFIGS)/esm2_35m.yaml \
		--set probe.epochs=10 --set probe.control_seeds=1
	$(PY) -m psl_probe.cli.run_analysis --config $(CONFIGS)/esm2_35m.yaml --skip-attention
	$(PY) -m psl_probe.cli.make_figures --config $(CONFIGS)/esm2_35m.yaml
	$(PY) -m psl_probe.cli.make_tables --config $(CONFIGS)/esm2_35m.yaml

# Scale comparison. Assumes `make data` has already run.
scales:
	@for cfg in $(CONFIGS)/esm2_35m.yaml $(CONFIGS)/esm2_150m.yaml $(CONFIGS)/esm2_650m.yaml; do \
		echo "=== $$cfg ==="; \
		$(PY) -m psl_probe.cli.extract_embeddings --config $$cfg $(EXTRA) && \
		$(PY) -m psl_probe.cli.run_probes --config $$cfg --seeds $(SEEDS) $(EXTRA) || exit 1; \
	done

test:
	$(PY) -m pytest tests/ -v

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check psl_probe tests || \
		echo "ruff not installed; skipping (pip install ruff)"

# Deletes derived results but never the raw Swiss-Prot download.
clean-results:
	rm -rf ../results/embeddings/*.h5 ../results/probes/*.csv \
	       ../results/analysis/* ../visualization/figures/*.pdf \
	       ../visualization/figures/*.png
