"""Training utilities: seed/device helpers, metrics, train/eval loop, fit().

Model-agnostic — `fit` works on any nn.Module with forward(x) -> (B, 1).
"""

import os
import random

import numpy as np
import torch
from scipy.stats import spearmanr


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = float(np.abs(y_true - y_pred).mean())
    mse = float(((y_true - y_pred) ** 2).mean())
    if y_true.size < 2 or np.std(y_pred) == 0 or np.std(y_true) == 0:
        spearman = float("nan")
    else:
        rho, _ = spearmanr(y_true, y_pred)
        # round to 10 d.p. to absorb float32 rank-tie rounding (e.g. 0.9999…999 → 1.0)
        spearman = round(float(rho), 10)
    return {"mae": mae, "mse": mse, "spearman": spearman}
