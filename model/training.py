"""Training utilities: seed/device helpers, metrics, train/eval loop, fit().

Model-agnostic — `fit` works on any nn.Module with forward(x) -> (B, 1).
"""

import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).view(-1, 1).float()
        assert y.shape[0] == x.shape[0], (
            f"y batch size mismatch: y={y.shape[0]} vs x={x.shape[0]}"
        )
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        # skip backward/step if all params are frozen (e.g. early-stopping test with a fully-frozen model)
        if any(p.requires_grad for p in model.parameters()):
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / max(n_samples, 1)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy().flatten()
            preds.append(pred)
            trues.append(y.cpu().numpy().flatten())
    if not preds:
        raise ValueError("evaluate called on an empty DataLoader")
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return compute_metrics(y_true, y_pred)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    log_every: int = 1,
) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    best_state: dict | None = None
    best_val_mae = float("inf")
    best_metrics: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_spearman": val_metrics["spearman"],
        }
        history.append(row)

        improved = val_metrics["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {**val_metrics, "epoch": epoch}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % log_every == 0:
            star = " *" if improved else ""
            print(
                f"epoch {epoch:3d} | train_loss={train_loss:.4f} "
                f"val_mae={val_metrics['mae']:.4f} val_mse={val_metrics['mse']:.4f} "
                f"val_spearman={val_metrics['spearman']:.3f}{star}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_metrics": best_metrics, "best_state": best_state}
