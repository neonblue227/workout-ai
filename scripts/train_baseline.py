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
