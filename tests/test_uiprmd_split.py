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
