"""
Keypoint extraction utility.

Extracts and normalizes keypoints from MediaPipe Holistic results
for pose, face, and hand landmarks.
"""

from typing import Optional


def extract_pose_keypoints(results, frame_shape: tuple) -> Optional[dict[str, dict]]:
    """
    Extract pose keypoints from MediaPipe Holistic results.

    Args:
        results: MediaPipe Holistic processing results
        frame_shape: Tuple of (height, width) of the frame

    Returns:
        dict: Pose keypoints as {idx: {x, y, z, visibility}} or None if not detected
    """
    if not results.pose_landmarks:
        return None

    h, w = frame_shape[:2]
    keypoints = {}

    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        keypoints[str(idx)] = {
            "x": round(landmark.x * w, 2),
            "y": round(landmark.y * h, 2),
            "z": round(landmark.z * w, 2),  # z is relative to x
            "visibility": round(getattr(landmark, "visibility", 1.0), 4),
        }

    return keypoints


def extract_face_keypoints(results, frame_shape: tuple) -> Optional[dict[str, dict]]:
    """
    Extract face mesh keypoints from MediaPipe Holistic results.

    Args:
        results: MediaPipe Holistic processing results
        frame_shape: Tuple of (height, width) of the frame

    Returns:
        dict: Face keypoints as {idx: {x, y, z}} or None if not detected
    """
    if not results.face_landmarks:
        return None

    h, w = frame_shape[:2]
    keypoints = {}

    for idx, landmark in enumerate(results.face_landmarks.landmark):
        keypoints[str(idx)] = {
            "x": round(landmark.x * w, 2),
            "y": round(landmark.y * h, 2),
            "z": round(landmark.z * w, 2),
        }

    return keypoints


def extract_hand_keypoints(landmarks, frame_shape: tuple) -> Optional[dict[str, dict]]:
    """
    Extract hand keypoints from MediaPipe hand landmarks.

    Args:
        landmarks: MediaPipe hand landmarks object (left or right hand)
        frame_shape: Tuple of (height, width) of the frame

    Returns:
        dict: Hand keypoints as {idx: {x, y, z}} or None if not detected
    """
    if not landmarks:
        return None

    h, w = frame_shape[:2]
    keypoints = {}

    for idx, landmark in enumerate(landmarks.landmark):
        keypoints[str(idx)] = {
            "x": round(landmark.x * w, 2),
            "y": round(landmark.y * h, 2),
            "z": round(landmark.z * w, 2),
        }

    return keypoints


def extract_all_keypoints(results, frame_shape: tuple) -> dict:
    """
    Extract all keypoints from MediaPipe Holistic results.

    Combines pose, face, left hand, and right hand keypoints into
    a single dictionary. Undetected body parts have None values.

    Args:
        results: MediaPipe Holistic processing results
        frame_shape: Tuple of (height, width) of the frame

    Returns:
        dict: All keypoints with structure:
            {
                "pose": {...} or None,
                "face": {...} or None,
                "left_hand": {...} or None,
                "right_hand": {...} or None
            }
    """
    return {
        "pose": extract_pose_keypoints(results, frame_shape),
        "face": extract_face_keypoints(results, frame_shape),
        "left_hand": extract_hand_keypoints(results.left_hand_landmarks, frame_shape),
        "right_hand": extract_hand_keypoints(results.right_hand_landmarks, frame_shape),
    }
