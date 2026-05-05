# Training Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch LSTM training driver for the UI-PRMD pickle dataset that produces a baseline number on Ex1 (deep squat) + Ex5 (sit-to-stand) with CLI knobs for temporal-length and joint-subset ablations.

**Architecture:** Three new files + edits to two existing files. `model/lstm_torch.py` defines the `nn.Module`. `model/training.py` provides device/seed helpers, metrics, and a model-agnostic `fit()` loop with early stopping. `scripts/train_baseline.py` is the CLI glue that loads the pickle, splits 64/16/20 train/val/test, fits, and writes per-run artifacts (config.json, metrics.json, history.csv, weights.pt) to `model/runs/<run-name>/`.

**Tech Stack:** Python 3.12, PyTorch ≥2.4 (MPS on Apple Silicon), scipy (Spearman ρ), numpy, pickle, argparse, csv. Tests use pytest. Reference spec: `docs/superpowers/specs/2026-05-05-training-driver-design.md`.

---

## File Structure

**Create:**
- `model/lstm_torch.py` — `LSTMScorer(nn.Module)` regressor (~60 LOC)
- `model/training.py` — seed/device/metrics/loop helpers (~140 LOC)
- `scripts/train_baseline.py` — CLI driver (~180 LOC)
- `tests/__init__.py` — empty marker
- `tests/test_uiprmd_split.py` — split-function tests
- `tests/test_lstm_torch.py` — model shape / forward-pass tests
- `tests/test_training.py` — seed determinism, metrics, fit-loop tests

**Modify:**
- `model/uiprmd_pickle_dataset.py` — add `train_val_test_split()` (~30 LOC added)
- `pyproject.toml` — add `torch`, `scipy`, dev-dep `pytest`
- `docs/progression.md` — per CLAUDE.md mandatory workflow

**Per-run output dir** (created by driver, not by hand): `model/runs/<run-name>/{config.json, metrics.json, history.csv, weights.pt}`.

---

### Task 1: Add dependencies and pytest scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add torch + scipy to dependencies, pytest to dev-dependencies**

Edit `pyproject.toml` to look like this:
```toml
[project]
name = "workout-ai"
version = "0.1.0"
description = "Move-UP — gamified rehab/exercise companion (NSC 2026)"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "mediapipe==0.10.14",
    "opencv-python>=4.13.0.92",
    "kaggle>=1.6.0",
    "numpy>=1.26",
    "torch>=2.4.0",
    "scipy>=1.11",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Sync deps**

Run: `uv sync`
Expected: torch + scipy + pytest installed without error. PyTorch is ~600 MB; takes a minute on first install.

- [ ] **Step 3: Create empty tests dir marker**

Write `tests/__init__.py` with empty content (just so Python treats it as a package).

- [ ] **Step 4: Verify pytest finds tests dir**

Run: `uv run pytest --collect-only`
Expected: `no tests ran` or zero tests collected — but no errors.

- [ ] **Step 5: Verify torch + MPS**

Run: `uv run python -c "import torch; print(torch.__version__); print('mps:', torch.backends.mps.is_available())"`
Expected: a version string ≥ 2.4 and `mps: True`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py
git commit -m "chore: add torch, scipy, pytest deps for training driver"
```

---

### Task 2: Add `train_val_test_split` to dataset module

**Files:**
- Modify: `model/uiprmd_pickle_dataset.py`
- Create: `tests/test_uiprmd_split.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uiprmd_split.py`:
```python
import numpy as np
import pytest

from model.uiprmd_pickle_dataset import train_val_test_split


def _fake_data(n=1000, seq=50, feat=117, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, seq, feat)).astype(np.float32)
    y = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
    stratify = rng.integers(0, n_classes, size=n)
    return X, y, stratify


def test_split_sizes_default():
    X, y, s = _fake_data(n=1000)
    Xtr, Xv, Xte, ytr, yv, yte = train_val_test_split(X, y, stratify=s)
    assert Xtr.shape[0] + Xv.shape[0] + Xte.shape[0] == 1000
    # 20% test of 1000 = 200; 20% val of remaining 800 = 160; train = 640
    assert Xte.shape[0] == 200
    assert Xv.shape[0] == 160
    assert Xtr.shape[0] == 640


def test_split_y_aligned():
    X, y, s = _fake_data(n=200)
    Xtr, Xv, Xte, ytr, yv, yte = train_val_test_split(X, y, stratify=s)
    assert ytr.shape[0] == Xtr.shape[0]
    assert yv.shape[0] == Xv.shape[0]
    assert yte.shape[0] == Xte.shape[0]


def test_split_no_index_overlap():
    X, y, s = _fake_data(n=300)
    # Use unique X values to detect overlap by content
    X = np.arange(300 * 50 * 117, dtype=np.float32).reshape(300, 50, 117)
    Xtr, Xv, Xte, ytr, yv, yte = train_val_test_split(X, y, stratify=s)
    tr_ids = {int(row.flat[0]) for row in Xtr}
    v_ids = {int(row.flat[0]) for row in Xv}
    te_ids = {int(row.flat[0]) for row in Xte}
    assert tr_ids.isdisjoint(v_ids)
    assert tr_ids.isdisjoint(te_ids)
    assert v_ids.isdisjoint(te_ids)
    assert len(tr_ids) + len(v_ids) + len(te_ids) == 300


def test_split_deterministic_with_seed():
    X, y, s = _fake_data(n=500, seed=1)
    a = train_val_test_split(X, y, stratify=s, seed=42)
    b = train_val_test_split(X, y, stratify=s, seed=42)
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_split_stratification_preserves_class_ratios():
    X, y, s = _fake_data(n=1000, n_classes=2)
    overall_ratio = (s == 0).mean()
    _, _, _, _, _, _ = train_val_test_split(X, y, stratify=s)  # smoke
    # We can't read indices back from the function, but we can check the
    # stratify proxy via labels: feed stratify==y_int back through.
    s_as_y = s.astype(np.float32)
    _, _, _, str_tr, str_v, str_te = train_val_test_split(X, s_as_y, stratify=s)
    # Each split should have a class-0 ratio within 5pp of overall
    for arr in (str_tr, str_v, str_te):
        assert abs((arr == 0).mean() - overall_ratio) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_uiprmd_split.py -v`
Expected: ImportError or AttributeError because `train_val_test_split` doesn't exist yet.

- [ ] **Step 3: Add the function**

Append to `model/uiprmd_pickle_dataset.py` (after the existing `train_test_split` function):
```python
def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    seed: int = 42,
    stratify: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Three-way split: peel test first, then peel val from remaining train pool.
    val_frac is applied to the post-test pool (so 0.2/0.2 -> 64/16/20)."""
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_frac=test_frac, seed=seed, stratify=stratify,
    )
    if stratify is not None:
        # Recover the stratify labels for the pool by re-running the same split
        # on stratify with the same seed.
        s_pool, _, _, _ = train_test_split(
            stratify, stratify, test_frac=test_frac, seed=seed, stratify=stratify,
        )
        stratify_pool = s_pool
    else:
        stratify_pool = None
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_frac=val_frac, seed=seed + 1, stratify=stratify_pool,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_uiprmd_split.py -v`
Expected: 5 tests passed.

- [ ] **Step 5: Commit**

```bash
git add model/uiprmd_pickle_dataset.py tests/test_uiprmd_split.py
git commit -m "feat(dataset): add train_val_test_split with stratification"
```

---

### Task 3: PyTorch LSTM model

**Files:**
- Create: `model/lstm_torch.py`
- Create: `tests/test_lstm_torch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lstm_torch.py`:
```python
import torch

from model.lstm_torch import LSTMScorer


def test_forward_shape_default():
    m = LSTMScorer(num_features=117)
    x = torch.randn(8, 50, 117)
    y = m(x)
    assert y.shape == (8, 1)


def test_forward_output_in_unit_interval():
    m = LSTMScorer(num_features=117)
    m.eval()
    x = torch.randn(4, 50, 117) * 5  # large inputs
    with torch.no_grad():
        y = m(x)
    assert (y >= 0).all() and (y <= 1).all(), f"sigmoid violated: {y}"


def test_configurable_features_and_hidden():
    m = LSTMScorer(num_features=60, hidden_1=32, hidden_2=16, dropout=0.0)
    x = torch.randn(2, 25, 60)
    y = m(x)
    assert y.shape == (2, 1)


def test_param_count_matches_config():
    m1 = LSTMScorer(num_features=117, hidden_1=64, hidden_2=32)
    m2 = LSTMScorer(num_features=117, hidden_1=128, hidden_2=64)
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    assert n2 > n1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lstm_torch.py -v`
Expected: ImportError because `model/lstm_torch.py` doesn't exist.

- [ ] **Step 3: Implement the model**

Create `model/lstm_torch.py`:
```python
"""PyTorch LSTM regressor for UI-PRMD posture-quality scoring (PoC)."""

import torch
import torch.nn as nn


class LSTMScorer(nn.Module):
    def __init__(
        self,
        num_features: int = 117,
        hidden_1: int = 64,
        hidden_2: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=num_features, hidden_size=hidden_1, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_1, hidden_size=hidden_2, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, (h_n, _) = self.lstm2(out)
        last = h_n[-1]
        last = torch.relu(self.fc1(last))
        return torch.sigmoid(self.fc2(last))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lstm_torch.py -v`
Expected: 4 tests passed.

- [ ] **Step 5: Commit**

```bash
git add model/lstm_torch.py tests/test_lstm_torch.py
git commit -m "feat(model): add PyTorch LSTM regressor for posture scoring"
```

---

### Task 4: Seed and device helpers in `training.py`

**Files:**
- Create: `model/training.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training.py`:
```python
import random
import numpy as np
import torch

from model.training import set_seed, make_device


def test_set_seed_makes_random_deterministic():
    set_seed(123)
    a_py = random.random()
    a_np = np.random.rand()
    a_t = torch.rand(1).item()

    set_seed(123)
    b_py = random.random()
    b_np = np.random.rand()
    b_t = torch.rand(1).item()

    assert a_py == b_py
    assert a_np == b_np
    assert a_t == b_t


def test_make_device_returns_torch_device():
    d = make_device()
    assert isinstance(d, torch.device)
    assert d.type in {"mps", "cpu", "cuda"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py -v`
Expected: ImportError because `model/training.py` doesn't exist.

- [ ] **Step 3: Implement seed + device helpers**

Create `model/training.py`:
```python
"""Training utilities: seed/device helpers, metrics, train/eval loop, fit().

Model-agnostic — `fit` works on any nn.Module with forward(x) -> (B, 1).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v`
Expected: 2 tests passed.

- [ ] **Step 5: Commit**

```bash
git add model/training.py tests/test_training.py
git commit -m "feat(training): add seed and device helpers"
```

---

### Task 5: Metrics (MAE, MSE, Spearman) in `training.py`

**Files:**
- Modify: `model/training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_training.py`:
```python
def test_compute_metrics_perfect_match():
    from model.training import compute_metrics
    y_true = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    y_pred = y_true.copy()
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == 0.0
    assert m["mse"] == 0.0
    assert m["spearman"] == 1.0


def test_compute_metrics_inverse_predictions_negative_spearman():
    from model.training import compute_metrics
    y_true = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    y_pred = y_true[::-1].copy()
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] > 0
    assert m["spearman"] == -1.0


def test_compute_metrics_known_values():
    from model.training import compute_metrics
    y_true = np.array([0.0, 1.0], dtype=np.float32)
    y_pred = np.array([0.5, 0.5], dtype=np.float32)
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == 0.5
    assert m["mse"] == 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py -v`
Expected: 3 new tests fail with ImportError on `compute_metrics`.

- [ ] **Step 3: Implement `compute_metrics`**

Append to `model/training.py`:
```python
from scipy.stats import spearmanr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = float(np.abs(y_true - y_pred).mean())
    mse = float(((y_true - y_pred) ** 2).mean())
    if y_true.size < 2 or np.std(y_pred) == 0:
        spearman = float("nan")
    else:
        rho, _ = spearmanr(y_true, y_pred)
        spearman = float(rho)
    return {"mae": mae, "mse": mse, "spearman": spearman}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v`
Expected: 5 tests passed.

- [ ] **Step 5: Commit**

```bash
git add model/training.py tests/test_training.py
git commit -m "feat(training): add MAE/MSE/Spearman metrics"
```

---

### Task 6: `train_one_epoch` and `evaluate`

**Files:**
- Modify: `model/training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_training.py`:
```python
import torch
from torch.utils.data import DataLoader, TensorDataset


def _toy_loader(n=64, seq=10, feat=8, batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, seq, feat, generator=g)
    y = torch.rand(n, generator=g)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch, shuffle=False)


class _TinyRegressor(torch.nn.Module):
    def __init__(self, feat=8):
        super().__init__()
        self.lin = torch.nn.Linear(feat, 1)

    def forward(self, x):
        return torch.sigmoid(self.lin(x.mean(dim=1)))


def test_train_one_epoch_returns_float_loss():
    from model.training import train_one_epoch
    model = _TinyRegressor()
    loader = _toy_loader()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    loss = train_one_epoch(model, loader, opt, loss_fn, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss >= 0


def test_train_one_epoch_decreases_loss_after_steps():
    from model.training import train_one_epoch
    torch.manual_seed(0)
    model = _TinyRegressor()
    loader = _toy_loader()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    losses = [
        train_one_epoch(model, loader, opt, loss_fn, torch.device("cpu"))
        for _ in range(20)
    ]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_evaluate_returns_metrics_dict():
    from model.training import evaluate
    model = _TinyRegressor()
    loader = _toy_loader(n=32)
    m = evaluate(model, loader, torch.device("cpu"))
    assert {"mae", "mse", "spearman"} <= set(m.keys())
    assert isinstance(m["mae"], float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py -v`
Expected: 3 new tests fail with ImportError.

- [ ] **Step 3: Implement loop and eval**

Append to `model/training.py`:
```python
import torch.nn as nn
from torch.utils.data import DataLoader


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
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
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
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return compute_metrics(y_true, y_pred)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v`
Expected: 8 tests passed.

- [ ] **Step 5: Commit**

```bash
git add model/training.py tests/test_training.py
git commit -m "feat(training): add train_one_epoch and evaluate"
```

---

### Task 7: `fit` with early stopping + best-state restore

**Files:**
- Modify: `model/training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_training.py`:
```python
def test_fit_returns_history_and_best_state():
    from model.training import fit
    torch.manual_seed(0)
    model = _TinyRegressor()
    train_loader = _toy_loader(n=64)
    val_loader = _toy_loader(n=32, seed=1)
    result = fit(
        model, train_loader, val_loader,
        epochs=5, lr=1e-2, patience=10,
        device=torch.device("cpu"),
    )
    assert "history" in result and "best_metrics" in result and "best_state" in result
    h = result["history"]
    assert len(h) == 5
    for row in h:
        assert {"epoch", "train_loss", "val_mae", "val_mse", "val_spearman"} <= set(row)


def test_fit_early_stopping_triggers_when_no_improvement():
    """A frozen model never improves; with patience=2 and epochs=20, fit
    should stop within ~3 epochs."""
    from model.training import fit
    torch.manual_seed(0)
    model = _TinyRegressor()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen — no improvement possible
    train_loader = _toy_loader()
    val_loader = _toy_loader(n=32, seed=1)
    result = fit(
        model, train_loader, val_loader,
        epochs=20, lr=1e-2, patience=2,
        device=torch.device("cpu"),
    )
    assert len(result["history"]) <= 5, f"early stop did not trigger: {len(result['history'])} epochs"


def test_fit_restores_best_weights():
    """Best state dict should match the state at the lowest val_mae epoch."""
    from model.training import fit
    torch.manual_seed(0)
    model = _TinyRegressor()
    train_loader = _toy_loader()
    val_loader = _toy_loader(n=32, seed=1)
    result = fit(
        model, train_loader, val_loader,
        epochs=10, lr=1e-2, patience=20,
        device=torch.device("cpu"),
    )
    best_epoch = min(result["history"], key=lambda r: r["val_mae"])["epoch"]
    assert result["best_metrics"]["epoch"] == best_epoch
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py -v`
Expected: 3 new tests fail.

- [ ] **Step 3: Implement `fit`**

Append to `model/training.py`:
```python
import copy


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
    best_epoch = -1
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
            best_epoch = epoch
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v`
Expected: 11 tests passed.

- [ ] **Step 5: Commit**

```bash
git add model/training.py tests/test_training.py
git commit -m "feat(training): add fit() with early stopping and best-state restore"
```

---

### Task 8: CLI skeleton for `train_baseline.py`

**Files:**
- Create: `scripts/train_baseline.py`

- [ ] **Step 1: Implement the CLI parser and `main()` skeleton**

Create `scripts/train_baseline.py`:
```python
"""Training driver for UI-PRMD pickle PoC baseline + ablations.

See docs/superpowers/specs/2026-05-05-training-driver-design.md for design.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

# Project root on path so model/* imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    # Data
    ap.add_argument("--exercises", choices=["squat", "sit_to_stand", "both"], default="both")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--joint-subset", choices=["all", "random", "top-variance"], default="all")
    ap.add_argument("--joint-subset-size", type=int, default=39)
    ap.add_argument("--joint-subset-seed", type=int, default=0)
    # Training
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hidden-1", type=int, default=64)
    ap.add_argument("--hidden-2", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    # Run mgmt
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("model/runs"))
    ap.add_argument("--strict-deterministic", action="store_true")
    ap.add_argument("--no-save-weights", action="store_true")
    return ap.parse_args()


def auto_run_name(args: argparse.Namespace) -> str:
    today = date.today().isoformat()
    js = "all" if args.joint_subset == "all" else f"{args.joint_subset[:3]}{args.joint_subset_size}"
    return (
        f"{today}_{args.exercises}_seq{args.seq_len}_j{js}"
        f"_h{args.hidden_1}-{args.hidden_2}_s{args.seed}"
    )


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    run_name = args.run_name or auto_run_name(args)
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}")
    print(f"Out: {run_dir}")
    # Body wired in later tasks
    raise SystemExit("not yet implemented — see Task 9")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI parses and prints help**

Run: `uv run python scripts/train_baseline.py --help`
Expected: argparse usage block listing all flags. No errors.

- [ ] **Step 3: Verify the auto-run-name function**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
import train_baseline as tb
class A: exercises='both'; seq_len=50; joint_subset='all'; joint_subset_size=39; hidden_1=64; hidden_2=32; seed=42
print(tb.auto_run_name(A()))
"
```
Expected: a string like `2026-05-05_both_seq50_jall_h64-32_s42`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_baseline.py
git commit -m "feat(driver): add train_baseline.py CLI skeleton"
```

---

### Task 9: Wire data → split → model → fit in the driver

**Files:**
- Modify: `scripts/train_baseline.py`

- [ ] **Step 1: Implement the data-loading + training body**

Replace the body of `main()` in `scripts/train_baseline.py`. Final content of the file:
```python
"""Training driver for UI-PRMD pickle PoC baseline + ablations.

See docs/superpowers/specs/2026-05-05-training-driver-design.md for design.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Project root on path so model/* imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.lstm_torch import LSTMScorer  # type: ignore[import-not-found]
from model.training import (  # type: ignore[import-not-found]
    evaluate,
    fit,
    make_device,
    set_seed,
)
from model.uiprmd_pickle_dataset import (  # type: ignore[import-not-found]
    EXERCISES,
    load_combined,
    load_exercise,
    train_val_test_split,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exercises", choices=["squat", "sit_to_stand", "both"], default="both")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--joint-subset", choices=["all", "random", "top-variance"], default="all")
    ap.add_argument("--joint-subset-size", type=int, default=39)
    ap.add_argument("--joint-subset-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hidden-1", type=int, default=64)
    ap.add_argument("--hidden-2", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("model/runs"))
    ap.add_argument("--strict-deterministic", action="store_true")
    ap.add_argument("--no-save-weights", action="store_true")
    return ap.parse_args()


def auto_run_name(args: argparse.Namespace) -> str:
    today = date.today().isoformat()
    js = "all" if args.joint_subset == "all" else f"{args.joint_subset[:3]}{args.joint_subset_size}"
    return (
        f"{today}_{args.exercises}_seq{args.seq_len}_j{js}"
        f"_h{args.hidden_1}-{args.hidden_2}_s{args.seed}"
    )


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, stratify) reshaped as (N, T, J, 3) for joint-subset support."""
    if args.exercises == "both":
        # load_combined returns flattened (N, T, 117). We need the un-flattened
        # (N, T, 39, 3) shape for joint-subset slicing — re-load raw.
        import pickle
        from model.uiprmd_pickle_dataset import DEFAULT_PICKLE  # type: ignore[import-not-found]
        with open(DEFAULT_PICKLE, "rb") as f:
            d = pickle.load(f)
        Xs, ys, ids = [], [], []
        for ex_id, name in enumerate(EXERCISES):
            X, y = d[EXERCISES[name]]
            X = X.astype(np.float32)
            y = y.flatten().astype(np.float32)
            Xs.append(X)
            ys.append(y)
            ids.append(np.full(len(y), ex_id, dtype=np.int32))
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ids)
    else:
        # Single exercise — load_exercise returns (N, T, 117); we need (N, T, 39, 3)
        import pickle
        from model.uiprmd_pickle_dataset import DEFAULT_PICKLE  # type: ignore[import-not-found]
        with open(DEFAULT_PICKLE, "rb") as f:
            d = pickle.load(f)
        X, y = d[EXERCISES[args.exercises]]
        X = X.astype(np.float32)
        y = y.flatten().astype(np.float32)
        ids = np.zeros(len(y), dtype=np.int32)
        return X, y, ids


def apply_seq_len(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Slice the front of the temporal axis."""
    if seq_len > X.shape[1]:
        raise ValueError(f"seq_len={seq_len} > available frames ({X.shape[1]})")
    return X[:, :seq_len]


def select_joint_indices(X_train: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Pick joint indices (out of 39) per the joint-subset strategy.
    `top-variance` is computed on TRAIN ONLY to avoid leakage."""
    n_total = X_train.shape[2]
    if args.joint_subset == "all":
        return np.arange(n_total)
    size = min(args.joint_subset_size, n_total)
    if args.joint_subset == "random":
        rng = np.random.default_rng(args.joint_subset_seed)
        return np.sort(rng.permutation(n_total)[:size])
    # top-variance
    var = X_train.reshape(-1, n_total, X_train.shape[3]).var(axis=(0, 2))
    return np.sort(np.argsort(var)[-size:])


def slice_joints(X: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """X is (N, T, J, 3); return (N, T, S, 3) where S = len(joints)."""
    return X[:, :, joints, :]


def flatten_features(X: np.ndarray) -> np.ndarray:
    """(N, T, J, 3) -> (N, T, J*3)."""
    return X.reshape(X.shape[0], X.shape[1], -1)


def make_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or auto_run_name(args)
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, strict=args.strict_deterministic)
    device = make_device()
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Out: {run_dir}")

    X, y, ids = load_data(args)
    print(f"Loaded {X.shape[0]} clips, raw shape {X.shape}")

    X = apply_seq_len(X, args.seq_len)
    Xtr, Xv, Xte, ytr, yv, yte = train_val_test_split(
        X, y, stratify=ids, seed=args.seed,
    )
    print(f"Split: train={Xtr.shape[0]}  val={Xv.shape[0]}  test={Xte.shape[0]}")

    joints = select_joint_indices(Xtr, args)
    Xtr_s = slice_joints(Xtr, joints)
    Xv_s = slice_joints(Xv, joints)
    Xte_s = slice_joints(Xte, joints)
    Xtr_f = flatten_features(Xtr_s)
    Xv_f = flatten_features(Xv_s)
    Xte_f = flatten_features(Xte_s)
    num_features = Xtr_f.shape[2]
    print(f"Joints used: {len(joints)}  num_features: {num_features}")

    train_loader = make_loader(Xtr_f, ytr, args.batch_size, shuffle=True)
    val_loader = make_loader(Xv_f, yv, args.batch_size, shuffle=False)
    test_loader = make_loader(Xte_f, yte, args.batch_size, shuffle=False)

    model = LSTMScorer(
        num_features=num_features,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        dropout=args.dropout,
    )
    result = fit(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience, device=device,
    )

    test_metrics = evaluate(model, test_loader, device)
    print()
    print("=" * 60)
    print(f"  TEST    mae={test_metrics['mae']:.4f}  "
          f"mse={test_metrics['mse']:.4f}  "
          f"spearman={test_metrics['spearman']:.3f}")
    print(f"  BEST VAL ep={result['best_metrics'].get('epoch', '?')}  "
          f"mae={result['best_metrics'].get('mae', float('nan')):.4f}")
    print("=" * 60)

    # Save artifacts
    config = {
        **vars(args),
        "num_features": num_features,
        "joints_used": joints.tolist(),
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }
    config["out_dir"] = str(args.out_dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    metrics = {
        "test": test_metrics,
        "best_val": result["best_metrics"],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_mae", "val_mse", "val_spearman"]
        )
        w.writeheader()
        w.writerows(result["history"])

    if not args.no_save_weights and result["best_state"] is not None:
        torch.save(result["best_state"], run_dir / "weights.pt")

    print(f"\nArtifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI on a tiny config (1 epoch)**

Run:
```bash
uv run python scripts/train_baseline.py --epochs 1 --patience 100 --run-name _smoketest --out-dir /tmp/move-up-runs
```
Expected:
- Prints `Run: _smoketest`, `Device: mps`/`cpu`, split sizes, joints info, one epoch line, test summary.
- `/tmp/move-up-runs/_smoketest/` contains config.json, metrics.json, history.csv, weights.pt.
- No exceptions.

- [ ] **Step 3: Inspect the artifacts**

Run:
```bash
ls -la /tmp/move-up-runs/_smoketest/
cat /tmp/move-up-runs/_smoketest/metrics.json
cat /tmp/move-up-runs/_smoketest/history.csv
```
Expected:
- 4 files present (3 JSON/CSV + weights.pt).
- metrics.json has `"test": {...}` and `"best_val": {...}` blocks.
- history.csv has 1 data row + header.

- [ ] **Step 4: Clean up smoke-test artifacts**

Run: `rm -rf /tmp/move-up-runs`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_baseline.py
git commit -m "feat(driver): wire data, split, model, fit + artifact saving"
```

---

### Task 10: Acceptance run — full baseline

**Files:**
- (No code changes — runs the baseline end-to-end on real defaults)

- [ ] **Step 1: Run the full baseline**

Run:
```bash
uv run python scripts/train_baseline.py
```
Expected:
- Auto run-name like `2026-05-05_both_seq50_jall_h64-32_s42`.
- ~10-30 epochs (early stopping kicks in eventually).
- Per-epoch lines printed.
- Test metrics summary printed at the end.
- Run dir created under `model/runs/`.

- [ ] **Step 2: Verify acceptance criteria from the spec**

Spec §10 acceptance criteria:
1. ✅ End-to-end without errors → confirmed by Step 1.
2. ✅ Final test metrics printed + saved → check stdout + `metrics.json`.
3. **Determinism check (CPU)**: Run again with `--seed 42 --strict-deterministic` (forced CPU via `PYTORCH_ENABLE_MPS_FALLBACK=0` is not enough; for true bit-determinism, set `CUDA_VISIBLE_DEVICES=""` AND override `make_device` is non-trivial). For PoC, accept "metrics within 1e-4" rather than bit-identical on MPS, and verify by running twice with same seed:

```bash
uv run python scripts/train_baseline.py --seed 42 --run-name _det_a --epochs 5 --out-dir /tmp/det
uv run python scripts/train_baseline.py --seed 42 --run-name _det_b --epochs 5 --out-dir /tmp/det
diff /tmp/det/_det_a/metrics.json /tmp/det/_det_b/metrics.json
```
Expected: small differences only in float precision (acceptable on MPS) or zero diff on CPU.

4. **Knob check**: Run a non-default seq-len:
```bash
uv run python scripts/train_baseline.py --seq-len 25 --run-name _knob --epochs 5 --out-dir /tmp/knob
```
Expected: completes without error; `metrics.json` differs from baseline run; `config.json` has `"seq_len": 25`.

- [ ] **Step 3: Clean up scratch dirs**

Run: `rm -rf /tmp/det /tmp/knob`

- [ ] **Step 4: Commit the baseline run dir**

The baseline run is the reference — keep it for comparison with future ablation runs.

```bash
git add model/runs/
git commit -m "chore: capture baseline run artifacts (Ex1+Ex5, seq50, all joints)"
```

---

### Task 11: Update progression.md per CLAUDE.md mandatory workflow

**Files:**
- Modify: `docs/progression.md`

- [ ] **Step 1: Update the PoC subsection to reflect driver completion**

Edit `docs/progression.md`. Find the section titled "#### PoC Baseline Scaffolding (UI-PRMD via cpm121/uiprmd-full10ex) ✅" and add two new rows to its table for the training driver, then update the surrounding prose:

```markdown
| 🧠 PyTorch LSTM scorer | `nn.Module` regressor — input (B, 50, num_features), sigmoid output ∈ [0,1] | `model/lstm_torch.py` |
| 🏃 Training driver + ablation knobs | seed/device helpers, `fit()` with early stopping, MAE/MSE/Spearman, CLI for seq-len + joint-subset | `model/training.py`, `scripts/train_baseline.py` |
| 🧪 Unit tests | split, model shape, metrics, fit-loop, early stopping | `tests/test_*.py` |
```

Also update the "Dataset สรุป" line to add a note about the baseline:
```
**Baseline ที่บันทึกไว้**: ดู `model/runs/<run-name>/metrics.json` (`metrics.json` มีทั้ง test + best_val)
```

- [ ] **Step 2: Bump document version footer**

Find the line `*Document version: aligned with Master Plan v1.0 (2 พ.ค. 2026) — last update 5 พ.ค. 2026 (PoC scaffolding)*` and replace with:
```
*Document version: aligned with Master Plan v1.0 (2 พ.ค. 2026) — last update 5 พ.ค. 2026 (PoC training driver lands)*
```

- [ ] **Step 3: Commit**

```bash
git add docs/progression.md
git commit -m "docs: progression update for PoC training driver completion"
```

---

## Self-Review

**Spec coverage check** — every section of the spec mapped to a task:

| Spec section | Task |
|---|---|
| §3 Architecture (3 new files + edits) | All tasks 2-9 |
| §4 lstm_torch.py | Task 3 |
| §5 training.py — seed/device | Task 4 |
| §5 training.py — metrics | Task 5 |
| §5 training.py — train_one_epoch + evaluate | Task 6 |
| §5 training.py — fit + early stopping | Task 7 |
| §6 train_val_test_split | Task 2 |
| §7 train_baseline.py CLI + flags | Task 8 |
| §7 joint-subset implementation | Task 9 (`select_joint_indices` + `slice_joints`) |
| §7 seq-len slicing | Task 9 (`apply_seq_len`) |
| §7 auto run-name | Task 8 |
| §8 outputs (config/metrics/history/weights) | Task 9 |
| §9 dependencies | Task 1 |
| §10 acceptance criteria | Task 10 |
| §11 deferred work | Not in plan (correctly out of scope) |
| §12 risks | Mitigations baked into Task 4 (`strict` flag) and Task 10 (acceptance check accepts MPS float drift) |

**Placeholder scan**: no TBDs / TODOs / "implement later". Each step has executable content.

**Type consistency check**:
- `LSTMScorer(num_features, hidden_1, hidden_2, dropout)` signature consistent across Task 3 (definition), Task 9 (caller).
- `fit(model, train_loader, val_loader, *, epochs, lr, patience, device, log_every)` signature consistent across Task 7 (definition), Task 9 (caller).
- `train_val_test_split` signature consistent across Task 2 (definition), Task 9 (caller).
- `compute_metrics`, `evaluate`, `train_one_epoch` reused identically.

No gaps found.
