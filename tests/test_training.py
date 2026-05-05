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
