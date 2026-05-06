"""
Fetch UI-PRMD via the Kaggle mirror cpm121/uiprmd-full10ex.

This mirror ships a single pickle (uiprmd_all_exercises.pkl) with all 10
movements pre-segmented to (N, 50_frames, 39_joints, 3_xyz) plus continuous
quality scores. We don't need a labels.csv — labels are inside the pickle.

After this script, model/uiprmd_pickle_dataset.py exposes the loader.

Prereqs:
    1. uv add kaggle  (already done)
    2. <project>/.kaggle/kaggle.json with chmod 600
       (gitignored — kept project-local instead of ~/.kaggle/)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_SLUG = "cpm121/uiprmd-full10ex"
DEFAULT_TARGET_DIR = Path("model/data/uiprmd")
PICKLE_NAME = "uiprmd_all_exercises.pkl"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_KAGGLE_DIR = PROJECT_ROOT / ".kaggle"


def check_kaggle() -> None:
    """Verify the kaggle CLI + project-local credentials are present, and
    point the kaggle CLI at <project>/.kaggle/ via KAGGLE_CONFIG_DIR."""
    if shutil.which("kaggle") is None:
        sys.exit(
            "ERROR: kaggle CLI not found.\n"
            "  uv add kaggle    (or pip install kaggle)"
        )
    cred = PROJECT_KAGGLE_DIR / "kaggle.json"
    if not cred.exists():
        sys.exit(
            f"ERROR: Kaggle credentials not found at {cred}\n"
            "  1. https://www.kaggle.com/settings -> API -> Create New Token\n"
            f"  2. mkdir -p {PROJECT_KAGGLE_DIR} && mv ~/Downloads/kaggle.json {cred}\n"
            f"  3. chmod 600 {cred}"
        )
    # Tell the kaggle CLI to read from the project-local dir instead of ~/.kaggle/
    os.environ["KAGGLE_CONFIG_DIR"] = str(PROJECT_KAGGLE_DIR)


def download(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_SLUG} -> {target_dir}")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_SLUG,
            "-p", str(target_dir),
            "--unzip",
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target-dir", type=Path, default=DEFAULT_TARGET_DIR,
        help=f"Where to download (default: {DEFAULT_TARGET_DIR})",
    )
    ap.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, just summarize an already-downloaded pickle",
    )
    args = ap.parse_args()

    pkl = args.target_dir / PICKLE_NAME

    if not args.skip_download:
        check_kaggle()
        download(args.target_dir)

    if not pkl.exists():
        sys.exit(
            f"ERROR: expected {pkl} after download. Inspect with:\n"
            f"  ls -R {args.target_dir}"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.uiprmd_pickle_dataset import summarize  # type: ignore[import-not-found]
    summarize(pkl)

    print(
        "\nNext steps:\n"
        "  uv run python model/uiprmd_pickle_dataset.py\n"
        "    -> sanity-check load + stratified train/test split\n"
        "  Then point the training notebook at it.\n"
    )


if __name__ == "__main__":
    main()
