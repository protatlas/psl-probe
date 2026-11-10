"""Determinism helpers.

Full bit-wise determinism on GPU is only achievable at a throughput cost; we
therefore expose ``deterministic`` as an explicit flag and default it to ``False``
for embedding extraction (inference-only, numerically stable) and ``True`` for
probe fitting, where run-to-run variance would otherwise confound small effects.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Base seed applied to all RNGs.
    deterministic:
        If ``True``, enable deterministic cuDNN/cuBLAS kernels. This requires
        ``CUBLAS_WORKSPACE_CONFIG`` to be set before CUDA initialisation, which we
        do here defensively, and typically costs 10-30% throughput.

    Returns
    -------
    int
        The seed that was applied (returned for convenient logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("Seeded RNGs with seed=%d (deterministic=%s)", seed, deterministic)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give each :class:`~torch.utils.data.DataLoader` worker a distinct stream."""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
