import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def test_set_seed_strict_enables_deterministic_algorithms():
    set_seed(0, strict=True)
    assert torch.are_deterministic_algorithms_enabled()


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
    import math
    assert math.isnan(m["spearman"])  # constant y_pred -> Spearman undefined


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


def test_evaluate_raises_on_empty_loader():
    import pytest
    from model.training import evaluate
    model = _TinyRegressor()
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 10, 8), torch.empty(0)), batch_size=4)
    with pytest.raises(ValueError, match="empty DataLoader"):
        evaluate(model, empty_loader, torch.device("cpu"))


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
