"""Layer-wise embedding extraction with length-bucketed batching.

Throughput on a single 24 GB GPU is dominated by padding waste, because Swiss-Prot
lengths span two orders of magnitude. We therefore sort by length and form batches
under a *token* budget rather than a fixed sequence count, which keeps GPU memory
roughly constant and typically gives a 3-5x speed-up over naive fixed batching.

Pooled representations for every layer are written to a single HDF5 file with
datasets ``pooled/<strategy>`` of shape ``(n_proteins, n_layers, hidden)``. Storing
all layers at once means the expensive forward pass is paid exactly once for the
entire layer-wise study.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from psl_probe.config import Config
from psl_probe.embeddings.models import PLMWrapper
from psl_probe.embeddings.pooling import pool_representations
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def length_bucketed_batches(
    lengths: list[int], batch_tokens: int, max_batch_size: int = 128
) -> Iterator[list[int]]:
    """Yield index batches whose padded token count stays under ``batch_tokens``.

    Indices are ordered by length so that each batch contains similarly sized
    sequences; the padded cost of a batch is ``max_length * batch_size``.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch: list[int] = []
    longest = 0
    for index in order:
        candidate_longest = max(longest, lengths[index])
        if batch and (
            candidate_longest * (len(batch) + 1) > batch_tokens
            or len(batch) >= max_batch_size
        ):
            yield batch
            batch, longest = [index], lengths[index]
        else:
            batch.append(index)
            longest = candidate_longest
    if batch:
        yield batch


def extract_embeddings(
    dataset: pd.DataFrame,
    plm: PLMWrapper,
    cfg: Config,
    output_path: Path | str,
    overwrite: bool = False,
) -> Path:
    """Run the pLM over the benchmark and store pooled per-layer representations.

    Parameters
    ----------
    dataset:
        Benchmark table with ``accession``, ``sequence`` and ``length`` columns.
    plm:
        A loaded :class:`~psl_probe.embeddings.models.PLMWrapper`.
    cfg:
        Full configuration; ``cfg.pooling`` selects the pooling strategies.
    output_path:
        Destination HDF5 file.
    overwrite:
        Recompute even if ``output_path`` already exists.

    Returns
    -------
    pathlib.Path
        The written HDF5 file, containing:

        - ``accessions``  : ``(n,)`` UTF-8 accession strings, row order of all arrays
        - ``lengths``     : ``(n,)`` int32 residue counts (post-truncation)
        - ``pooled/<s>``  : ``(n, n_layers, hidden)`` float16 per strategy
        - attributes recording model id, dtype, truncation length and pooling windows
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        logger.info("Embeddings already exist at %s (use --overwrite to recompute)", output_path)
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sequences = dataset["sequence"].tolist()
    accessions = dataset["accession"].tolist()
    lengths = [min(len(s), cfg.model.max_length) for s in sequences]
    n_proteins = len(sequences)
    n_layers = plm.n_hidden_states
    hidden_size = plm.hidden_size
    strategies = cfg.pooling.strategies

    logger.info(
        "Extracting %s: %d proteins x %d layers x %d dims, strategies=%s",
        cfg.model.tag, n_proteins, n_layers, hidden_size, strategies,
    )

    tmp_path = output_path.with_suffix(".tmp.h5")
    with h5py.File(tmp_path, "w") as store:
        store.attrs["model_id"] = cfg.model.hf_id
        store.attrs["model_tag"] = cfg.model.tag
        store.attrs["family"] = cfg.model.family
        store.attrs["dtype"] = cfg.model.dtype
        store.attrs["max_length"] = cfg.model.max_length
        store.attrs["nterm_window"] = cfg.pooling.nterm_window
        store.attrs["cterm_window"] = cfg.pooling.cterm_window
        store.attrs["n_layers"] = n_layers

        store.create_dataset(
            "accessions", data=np.array(accessions, dtype=h5py.string_dtype("utf-8"))
        )
        store.create_dataset("lengths", data=np.array(lengths, dtype=np.int32))

        group = store.create_group("pooled")
        datasets = {
            strategy: group.create_dataset(
                strategy,
                shape=(n_proteins, n_layers, hidden_size),
                dtype=np.float16,
                # Chunk along proteins so probe training can stream one layer cheaply.
                chunks=(min(64, n_proteins), n_layers, hidden_size),
                compression="lzf",
            )
            for strategy in strategies
        }

        batches = list(length_bucketed_batches(lengths, cfg.model.batch_tokens))
        n_residues_seen = 0
        output = pooled = None
        for batch_indices in tqdm(batches, desc=f"embedding {cfg.model.tag}", unit="batch"):
            batch_sequences = [sequences[i] for i in batch_indices]
            # Drop the previous batch's activations before allocating the next
            # ones; otherwise two full sets of hidden states are live at once and
            # peak memory is ~2x what a single batch needs.
            del output, pooled
            output = pooled = None
            output = plm.encode(batch_sequences)
            pooled = pool_representations(
                output.hidden_states,
                output.mask,
                strategies=strategies,
                nterm_window=cfg.pooling.nterm_window,
                cterm_window=cfg.pooling.cterm_window,
            )
            # HDF5 fancy indexing requires monotonically increasing indices.
            order = np.argsort(batch_indices)
            sorted_indices = np.asarray(batch_indices)[order]
            for strategy, tensor in pooled.items():
                # (n_layers, batch, hidden) -> (batch, n_layers, hidden)
                array = tensor.permute(1, 0, 2).to(torch.float16).cpu().numpy()
                # Fail fast on non-finite values. A silently corrupted store only
                # surfaces much later as a NaN deep inside a sklearn metric, with
                # nothing in the traceback pointing back to extraction.
                if not np.isfinite(array).all():
                    n_bad = int((~np.isfinite(array)).sum())
                    worst = int(max(lengths[i] for i in batch_indices))
                    raise FloatingPointError(
                        f"Pooled '{strategy}' embeddings contain {n_bad} non-finite "
                        f"values (batch of {len(batch_indices)}, longest protein "
                        f"{worst} residues). Half-precision storage overflows when "
                        f"pooled magnitudes exceed 65504; re-run with "
                        f"model.dtype=float32 if this persists."
                    )
                datasets[strategy][sorted_indices] = array[order]
            n_residues_seen += int(output.mask.sum().item())

        store.attrs["n_residues"] = n_residues_seen

    tmp_path.replace(output_path)
    size_gb = output_path.stat().st_size / 1e9
    logger.info("Wrote %s (%.2f GB)", output_path, size_gb)
    return output_path


def load_pooled(
    path: Path | str, strategy: str, layer: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """Read pooled embeddings back as ``float32``.

    Parameters
    ----------
    path:
        HDF5 file written by :func:`extract_embeddings`.
    strategy:
        Pooling strategy name.
    layer:
        Layer index to read. ``None`` returns every layer.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        ``(features, accessions)`` with features of shape ``(n, hidden)`` when a
        layer is given and ``(n, n_layers, hidden)`` otherwise.
    """
    with h5py.File(path, "r") as store:
        accessions = [a.decode("utf-8") for a in store["accessions"][:]]
        dataset = store[f"pooled/{strategy}"]
        features = dataset[:, layer, :] if layer is not None else dataset[:]
    return np.asarray(features, dtype=np.float32), accessions


def embedding_metadata(path: Path | str) -> dict:
    """Return the provenance attributes of an embedding store."""
    with h5py.File(path, "r") as store:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in store.attrs.items()}
