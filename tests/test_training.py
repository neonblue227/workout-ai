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
