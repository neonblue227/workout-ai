"""
Loader for the cpm121/uiprmd-full10ex pickle (Vicon-positions UI-PRMD).

Pickle structure: dict {"Ex1"..."Ex10" -> (X, y)} where
  X : float32 (N_clips, 50_frames, 39_joints, 3_xyz)
  y : float32 (N_clips, 1)            continuous quality scores ~[0, 1]

We expose Ex1 (deep squat) and Ex5 (sit-to-stand) — the two NSC-scope
movements. Joints x XYZ are flattened (50, 39, 3) -> (50, 117) so the
existing LSTM in scoring_model.py reuses unchanged with num_features=117.
"""

import pickle
from pathlib import Path

import numpy as np

DEFAULT_PICKLE = Path("model/data/uiprmd/uiprmd_all_exercises.pkl")

EXERCISES = {
    "deep_squat": "Ex1",
    "sit_to_stand": "Ex5",
}

NUM_FEATURES = 39 * 3  # 117 (flattened joints x xyz)
SEQUENCE_LENGTH = 50


def load_exercise(
    name: str, pkl_path: Path = DEFAULT_PICKLE
) -> tuple[np.ndarray, np.ndarray]:
    """Load one exercise -> (X, y) with X shape (N, 50, 117), y shape (N,)."""
    if name not in EXERCISES:
        raise ValueError(
            f"Unknown exercise {name!r}; expected one of {list(EXERCISES)}"
        )
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    X, y = d[EXERCISES[name]]
    X = X.reshape(X.shape[0], X.shape[1], -1).astype(np.float32)
    y = y.flatten().astype(np.float32)
    return X, y


def load_combined(
    pkl_path: Path = DEFAULT_PICKLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Ex1 + Ex5 stacked. Returns (X, y, exercise_id) where exercise_id
    is 0 for deep_squat, 1 for sit_to_stand — useful for stratified splits."""
    Xs, ys, ids = [], [], []
    for ex_id, name in enumerate(EXERCISES):
        X, y = load_exercise(name, pkl_path)
        Xs.append(X)
        ys.append(y)
        ids.append(np.full(len(y), ex_id, dtype=np.int32))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ids)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.2,
    seed: int = 42,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random split. If stratify is given, preserves per-class frequencies.

    Use stratify=exercise_id when training on combined Ex1+Ex5 to keep
    both exercises proportionally represented in train/test.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if stratify is None:
        idx = rng.permutation(n)
        # round() avoids 1-sample undercount when class sizes aren't exact multiples of test_frac
        cut = round(n * test_frac)
        test_idx, train_idx = idx[:cut], idx[cut:]
    else:
        train_parts, test_parts = [], []
        for cls in np.unique(stratify):
            cls_idx = np.flatnonzero(stratify == cls)
            cls_idx = rng.permutation(cls_idx)
            cut = round(len(cls_idx) * test_frac)
            test_parts.append(cls_idx[:cut])
            train_parts.append(cls_idx[cut:])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


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


def summarize(pkl_path: Path = DEFAULT_PICKLE) -> None:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    print(f"Pickle: {pkl_path}")
    print(f"{'name':<14}{'kaggle_key':<6}{'X.shape':<24}{'y.min':>8}"
          f"{'y.max':>8}{'y.mean':>8}{'y.std':>8}")
    for name, key in EXERCISES.items():
        X, y = d[key]
        yy = y.flatten()
        print(f"{name:<14}{key:<6}{str(X.shape):<24}"
              f"{yy.min():>8.3f}{yy.max():>8.3f}"
              f"{yy.mean():>8.3f}{yy.std():>8.3f}")


if __name__ == "__main__":
    summarize()

    print("\nQuick load test:")
    X, y, ids = load_combined()
    print(f"  combined X: {X.shape}  y: {y.shape}  ids: {ids.shape}")
    print(f"  squat clips     : {(ids == 0).sum()}")
    print(f"  sit-to-stand    : {(ids == 1).sum()}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.2, stratify=ids)
    print(f"\nStratified 80/20 split:")
    print(f"  X_train: {Xtr.shape}   y_train: {ytr.shape}")
    print(f"  X_test : {Xte.shape}   y_test : {yte.shape}")
    print(f"  y_train mean={ytr.mean():.3f} std={ytr.std():.3f}")
    print(f"  y_test  mean={yte.mean():.3f} std={yte.std():.3f}")
