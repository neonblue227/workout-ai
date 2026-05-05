"""
NOT USED BY THE CURRENT POC — kept for reference.

Adapter: UI-PRMD Kinect skeleton txt -> MediaPipe-shaped JSON.

This module was written for raw-txt UI-PRMD mirrors (e.g. liza5757/uiprmd).
The mirror we settled on (cpm121/uiprmd-full10ex) ships a pre-processed
Vicon pickle instead — see model/uiprmd_pickle_dataset.py for the active
loader. Keep this file around in case we ever pull a raw-Kinect mirror.

UI-PRMD positions files store one frame per row, 25 joints x 3 coords = 75
columns (comma-separated ASCII).

feature_extractor.py expects MediaPipe-shaped pose data:
    {"frames": [{"pose": {"11": {"x":..., "y":..., "z":..., "visibility":...},
                          ...}}, ...]}

We emit that schema so PostureDataset works unchanged.

IMPORTANT: KINECT_JOINTS below assumes Microsoft Kinect v2 SDK joint ordering.
The UI-PRMD paper does not pin down column order verbatim, so before
trusting any features, run:

    uv run python model/kinect_to_features.py --inspect <one_positions.txt>

and confirm the joint XYZ values look sane (e.g. Head y > Hip y in a
standing frame).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KINECT_JOINTS = {
    "SpineBase": 0,
    "SpineMid": 1,
    "Neck": 2,
    "Head": 3,
    "ShoulderLeft": 4,
    "ElbowLeft": 5,
    "WristLeft": 6,
    "HandLeft": 7,
    "ShoulderRight": 8,
    "ElbowRight": 9,
    "WristRight": 10,
    "HandRight": 11,
    "HipLeft": 12,
    "KneeLeft": 13,
    "AnkleLeft": 14,
    "FootLeft": 15,
    "HipRight": 16,
    "KneeRight": 17,
    "AnkleRight": 18,
    "FootRight": 19,
    "SpineShoulder": 20,
}

# MediaPipe pose-landmark idx -> Kinect joint name. Only includes the indices
# that feature_extractor.ANGLE_DEFINITIONS and KEY_LANDMARKS actually read.
# Head is reused for nose/ear approximations because Kinect lacks face joints.
MEDIAPIPE_TO_KINECT = {
    0: "Head",
    7: "Head",
    8: "Head",
    11: "ShoulderLeft",
    12: "ShoulderRight",
    13: "ElbowLeft",
    14: "ElbowRight",
    15: "WristLeft",
    16: "WristRight",
    23: "HipLeft",
    24: "HipRight",
    25: "KneeLeft",
    26: "KneeRight",
    27: "AnkleLeft",
    28: "AnkleRight",
}


def load_kinect_positions(txt_path: Path) -> np.ndarray:
    """Load UI-PRMD positions txt -> (num_frames, 25, 3) array."""
    arr = np.loadtxt(txt_path, delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    n_frames, n_cols = arr.shape
    expected = 25 * 3
    if n_cols != expected:
        raise ValueError(
            f"{txt_path}: expected {expected} columns (25 joints * 3), got {n_cols}. "
            f"Run with --inspect to see the file shape, then adjust this loader."
        )
    return arr.reshape(n_frames, 25, 3)


def kinect_frame_to_mediapipe_pose(joints: np.ndarray) -> dict:
    pose = {}
    for mp_idx, kinect_name in MEDIAPIPE_TO_KINECT.items():
        x, y, z = joints[KINECT_JOINTS[kinect_name]]
        pose[str(mp_idx)] = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "visibility": 1.0,
        }
    return pose


def kinect_file_to_json(txt_path: Path, out_path: Path, fps: float = 30.0) -> None:
    frames_arr = load_kinect_positions(txt_path)
    frames = [
        {
            "frame": i,
            "timestamp": i / fps,
            "pose": kinect_frame_to_mediapipe_pose(joints),
            "face": None,
            "left_hand": None,
            "right_hand": None,
        }
        for i, joints in enumerate(frames_arr)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {
                "metadata": {
                    "source": str(txt_path),
                    "schema": "ui-prmd-kinect-25-as-mediapipe-33",
                    "num_frames": len(frames),
                    "fps": fps,
                },
                "frames": frames,
            },
            fh,
        )


def convert_dataset(uiprmd_dir: Path, out_dir: Path) -> int:
    keypoint_dir = out_dir / "keypoint"
    keypoint_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(uiprmd_dir.rglob("m0[15]_*positions*.txt"))
    if not txt_files:
        sys.exit(
            f"ERROR: no m01/m05 positions txt files under {uiprmd_dir}.\n"
            "Did you run scripts/fetch_uiprmd.py first?"
        )

    for txt in txt_files:
        kinect_file_to_json(txt, keypoint_dir / f"{txt.stem}.json")

    print(f"Converted {len(txt_files)} files -> {keypoint_dir}")
    return len(txt_files)


def inspect(txt_path: Path) -> None:
    arr = np.loadtxt(txt_path, delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    print(f"File: {txt_path}")
    print(f"  Shape: {arr.shape}  (frames x cols)")
    print(f"  First-frame head: {arr[0, :9]}")
    if arr.shape[1] != 75:
        print(f"  WARNING: expected 75 cols, got {arr.shape[1]}.")
        return
    joints = arr[0].reshape(25, 3)
    print("  First frame, first 5 joints (assuming Kinect v2 SDK order):")
    for i in range(5):
        name = next((n for n, idx in KINECT_JOINTS.items() if idx == i), f"j{i}")
        print(f"    [{i}] {name:<14}: {joints[i]}")
    print("  Sanity check (standing pose): Head.y should be greater than HipLeft.y")
    print(f"    Head.y     = {joints[KINECT_JOINTS['Head']][1]:.3f}")
    print(f"    HipLeft.y  = {joints[KINECT_JOINTS['HipLeft']][1]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "uiprmd_dir", type=Path, nargs="?",
        help="UI-PRMD root (target-dir from fetch_uiprmd.py)",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir for keypoint/ JSON (defaults to uiprmd_dir)",
    )
    ap.add_argument(
        "--inspect", type=Path, default=None,
        help="Inspect one positions.txt and exit (verify joint ordering)",
    )
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    if args.uiprmd_dir is None:
        ap.error("uiprmd_dir is required unless --inspect is used")

    out_dir = args.out_dir or args.uiprmd_dir
    convert_dataset(args.uiprmd_dir, out_dir)


if __name__ == "__main__":
    main()
