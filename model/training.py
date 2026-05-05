"""Training utilities: seed/device helpers, metrics, train/eval loop, fit().

Model-agnostic — `fit` works on any nn.Module with forward(x) -> (B, 1).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, strict: bool = False) -> None:
    """Seed RNGs for reproducibility across random / numpy / torch / MPS / CUDA.

    `strict=True` enables a best-effort deterministic mode (sets
    `torch.use_deterministic_algorithms(True, warn_only=True)` and CUBLAS env).
    Some ops on MPS lack deterministic kernels and will WARN, not raise — so
    bit-identical results aren't guaranteed even in strict mode. Strict mode
    also persists for the lifetime of the process; calling `set_seed(...)`
    without strict afterward does not undo it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # explicit for clarity; torch.manual_seed already seeds MPS on torch >= 2.0
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
