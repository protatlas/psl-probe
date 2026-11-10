"""psl_probe: layer-wise probing of protein language models for subcellular localization.

The package is organised into four stages that map one-to-one onto the CLI entry
points in :mod:`psl_probe.cli`:

1. ``data``       -- curate a Swiss-Prot derived, homology-partitioned benchmark.
2. ``embeddings`` -- extract per-layer, per-residue pLM representations.
3. ``probes``     -- fit linear / MLP probes and matched control tasks.
4. ``analysis``   -- selectivity, CKA, attention-signal alignment, ablations.
"""

import os as _os

# Use expandable CUDA segments unless the caller has chosen otherwise.
#
# Embedding extraction feeds the allocator a monotonically growing sequence of
# batch shapes (batches are length-bucketed, shortest first), so cached blocks
# from earlier batches are never the right size for later ones. With the default
# allocator this fragments badly: on an RTX 4090 extracting ESM-2 650M, peak
# *allocated* memory is 8.4 GB but peak *reserved* reaches 20.7 GB, which
# produces repeated allocation failures and a ~40% throughput loss even though
# the job comfortably fits. Expandable segments hold reserved at 8.6 GB.
#
# This must be set before the CUDA allocator initialises, which is why it lives
# at package import rather than in a CLI entry point.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"

__all__ = ["__version__"]
