"""Reproducibility helpers shared by all training entry points."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

_log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must be set before the first CUBLAS handle is created to take effect.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:  # torch < 1.11
            pass

    _log.info("Seed set to %d (deterministic=%s)", seed, deterministic)


def add_seed_args(parser) -> None:
    """Attach the standard --seed / --deterministic flags to an ArgumentParser."""
    parser.add_argument('--seed', type=int, default=42,
                        help="RNG seed for Python/NumPy/Torch (default: 42)")
    parser.add_argument('--deterministic', action='store_true',
                        help="Force deterministic cuDNN kernels. Slower; use for "
                             "runs whose numbers are reported.")
