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
    train_val_test_split,
)

# Map CLI exercise short-names to keys in model.uiprmd_pickle_dataset.EXERCISES
CLI_TO_EXERCISE = {
    "squat": "deep_squat",
    "sit_to_stand": "sit_to_stand",
}


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
        for ex_id, ex_name in enumerate(CLI_TO_EXERCISE.values()):
            X, y = d[EXERCISES[ex_name]]
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
        X, y = d[EXERCISES[CLI_TO_EXERCISE[args.exercises]]]
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
