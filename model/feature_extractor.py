"""
Feature Extractor for Posture Scoring.

Converts raw MediaPipe pose landmarks into invariant features suitable
for temporal learning: joint angles and normalized coordinates.
"""

import json
import os
import sys
from typing import Optional

import numpy as np

# Path setup for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.angle import angle_between_points
except ImportError:
    # Fallback implementation if import fails (e.g., running from Jupyter)
    from math import acos, degrees

    def angle_between_points(a, b, c):
        """
        Calculate angle (degrees) at point b formed by a-b-c.

        Args:
            a: First point as (x, y) tuple
            b: Vertex point as (x, y) tuple (angle is measured here)
            c: Third point as (x, y) tuple

        Returns:
            float: Angle in degrees, or None if calculation is not possible
        """
        ax, ay = a
        bx, by = b
        cx, cy = c
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        norm2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return None
        cosang = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return degrees(acos(cosang))


# MediaPipe Pose landmark indices
class PoseLandmark:
    """MediaPipe Pose landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Joint angle definitions: (point_a, vertex, point_c, name)
ANGLE_DEFINITIONS = [
    # Elbow angles
    (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_ELBOW,
        PoseLandmark.LEFT_WRIST,
        "L.Elbow",
    ),
    (
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.RIGHT_WRIST,
        "R.Elbow",
    ),
    # Knee angles
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE, "L.Knee"),
    (
        PoseLandmark.RIGHT_HIP,
        PoseLandmark.RIGHT_KNEE,
        PoseLandmark.RIGHT_ANKLE,
        "R.Knee",
    ),
    # Hip angles
    (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.LEFT_KNEE,
        "L.Hip",
    ),
    (
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_HIP,
        PoseLandmark.RIGHT_KNEE,
        "R.Hip",
    ),
    # Shoulder angles
    (
        PoseLandmark.LEFT_ELBOW,
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        "L.Shoulder",
    ),
    (
        PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_HIP,
        "R.Shoulder",
    ),
    # Neck angle (ear - shoulder - hip for head tilt)
    (
        PoseLandmark.LEFT_EAR,
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        "L.Neck",
    ),
    (
        PoseLandmark.RIGHT_EAR,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_HIP,
        "R.Neck",
    ),
]

# Key landmarks to normalize (upper body focus for stretching)
KEY_LANDMARKS = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
]


def get_landmark_position(pose_data: dict, idx: int) -> Optional[tuple]:
    """
    Get (x, y) position for a landmark index.

    Args:
        pose_data: Dict of {str(idx): {x, y, z, visibility}}
        idx: Landmark index

    Returns:
        (x, y) tuple or None if not found/low visibility
    """
    key = str(idx)
    if key not in pose_data:
        return None

    landmark = pose_data[key]
    visibility = landmark.get("visibility", 1.0)

    if visibility < 0.5:
        return None

    return (landmark["x"], landmark["y"])


def calculate_mid_hip(pose_data: dict) -> Optional[tuple]:
    """
    Calculate mid-hip position for normalization.

    Args:
        pose_data: Dict of pose landmarks

    Returns:
        (x, y) of mid-hip or None
    """
    left_hip = get_landmark_position(pose_data, PoseLandmark.LEFT_HIP)
    right_hip = get_landmark_position(pose_data, PoseLandmark.RIGHT_HIP)

    if left_hip is None or right_hip is None:
        return None

    return ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)


def calculate_torso_scale(pose_data: dict) -> float:
    """
    Calculate torso length for scale normalization.

    Uses distance from mid-shoulder to mid-hip.

    Args:
        pose_data: Dict of pose landmarks

    Returns:
        Torso length or 1.0 if cannot calculate
    """
    left_shoulder = get_landmark_position(pose_data, PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_landmark_position(pose_data, PoseLandmark.RIGHT_SHOULDER)
    mid_hip = calculate_mid_hip(pose_data)

    if left_shoulder is None or right_shoulder is None or mid_hip is None:
        return 1.0

    mid_shoulder = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )

    # Euclidean distance
    torso_length = (
        (mid_shoulder[0] - mid_hip[0]) ** 2 + (mid_shoulder[1] - mid_hip[1]) ** 2
    ) ** 0.5

    return max(torso_length, 1.0)  # Avoid division by zero


def extract_joint_angles(pose_data: dict) -> list:
    """
    Extract all joint angles from pose data.

    Args:
        pose_data: Dict of pose landmarks

    Returns:
        List of angles (normalized to 0-1 range), missing angles as 0.5
    """
    angles = []

    for a_idx, b_idx, c_idx, _ in ANGLE_DEFINITIONS:
        a = get_landmark_position(pose_data, a_idx)
        b = get_landmark_position(pose_data, b_idx)
        c = get_landmark_position(pose_data, c_idx)

        if a is None or b is None or c is None:
            angles.append(0.5)  # Neutral value for missing angles
        else:
            angle = angle_between_points(a, b, c)
            if angle is None:
                angles.append(0.5)
            else:
                # Normalize angle to 0-1 range (0-180 degrees)
                angles.append(angle / 180.0)

    return angles


def extract_normalized_coordinates(pose_data: dict) -> list:
    """
    Extract normalized coordinates relative to mid-hip.

    Coordinates are normalized by torso length for scale invariance.

    Args:
        pose_data: Dict of pose landmarks

    Returns:
        List of normalized (x, y) coordinates, flattened
    """
    mid_hip = calculate_mid_hip(pose_data)
    scale = calculate_torso_scale(pose_data)

    coords = []

    for idx in KEY_LANDMARKS:
        pos = get_landmark_position(pose_data, idx)

        if pos is None or mid_hip is None:
            coords.extend([0.0, 0.0])  # Centered default
        else:
            # Normalize relative to mid-hip and scale by torso length
            norm_x = (pos[0] - mid_hip[0]) / scale
            norm_y = (pos[1] - mid_hip[1]) / scale
            coords.extend([norm_x, norm_y])

    return coords


def extract_features_from_frame(frame_data: dict) -> Optional[np.ndarray]:
    """
    Extract feature vector from a single frame.

    Features = [joint_angles (10), normalized_coords (18)] = 28 features

    Args:
        frame_data: Frame dict with 'pose', 'face', 'left_hand', 'right_hand'

    Returns:
        numpy array of features or None if pose not detected
    """
    pose_data = frame_data.get("pose")

    if pose_data is None:
        return None

    # Extract features
    angles = extract_joint_angles(pose_data)  # 10 features
    coords = extract_normalized_coordinates(pose_data)  # 18 features (9 landmarks * 2)

    # Combine into feature vector
    features = angles + coords

    return np.array(features, dtype=np.float32)


def extract_features_from_file(json_path: str) -> np.ndarray:
    """
    Extract features from all frames in a JSON keypoint file.

    Args:
        json_path: Path to JSON keypoint file

    Returns:
        numpy array of shape (num_frames, num_features)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    features_list = []

    for frame in frames:
        features = extract_features_from_frame(frame)

        if features is not None:
            features_list.append(features)
        else:
            # Use zeros for frames without pose detection
            features_list.append(np.zeros(28, dtype=np.float32))

    if not features_list:
        return np.zeros((1, 28), dtype=np.float32)

    return np.array(features_list, dtype=np.float32)


def get_feature_names() -> list:
    """
    Get human-readable names for each feature.

    Returns:
        List of feature names
    """
    names = []

    # Angle names
    for _, _, _, name in ANGLE_DEFINITIONS:
        names.append(f"angle_{name}")

    # Coordinate names
    landmark_names = [
        "nose",
        "l_shoulder",
        "r_shoulder",
        "l_elbow",
        "r_elbow",
        "l_wrist",
        "r_wrist",
        "l_hip",
        "r_hip",
    ]
    for lm_name in landmark_names:
        names.append(f"norm_{lm_name}_x")
        names.append(f"norm_{lm_name}_y")

    return names


# Constants for external use
NUM_FEATURES = 28  # 10 angles + 18 normalized coords


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "model/data/neck_stretch/keypoint/L.json"

    print(f"Testing feature extraction on: {test_path}")
    features = extract_features_from_file(test_path)
    print(f"Output shape: {features.shape}")
    print(f"First frame features: {features[0]}")
    print(f"Feature names: {get_feature_names()}")
