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
