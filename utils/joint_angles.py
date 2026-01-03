"""
Joint angle calculation utility.

Calculates important joint angles (elbow, knee, hip) from pose keypoints.
"""

import mediapipe as mp

from .angle import angle_between_points

mp_holistic = mp.solutions.holistic


def calculate_joint_angles(keypoint_positions):
    """
    Calculate important joint angles for pose analysis.

    Calculates angles for:
        - Left/Right Elbow (Shoulder-Elbow-Wrist)
        - Left/Right Knee (Hip-Knee-Ankle)
        - Left/Right Hip (Shoulder-Hip-Knee)

    Args:
        keypoint_positions: Dict of {landmark_idx: (px, py, visibility)}

    Returns:
        dict: Joint angles as {name: angle_degrees}
    """
    angles = {}

    angle_definitions = [
        (
            mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
            mp_holistic.PoseLandmark.LEFT_ELBOW.value,
            mp_holistic.PoseLandmark.LEFT_WRIST.value,
            "L.Elbow",
        ),
        (
            mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
            mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
            mp_holistic.PoseLandmark.RIGHT_WRIST.value,
            "R.Elbow",
        ),
        (
            mp_holistic.PoseLandmark.LEFT_HIP.value,
            mp_holistic.PoseLandmark.LEFT_KNEE.value,
            mp_holistic.PoseLandmark.LEFT_ANKLE.value,
            "L.Knee",
        ),
        (
            mp_holistic.PoseLandmark.RIGHT_HIP.value,
            mp_holistic.PoseLandmark.RIGHT_KNEE.value,
            mp_holistic.PoseLandmark.RIGHT_ANKLE.value,
            "R.Knee",
        ),
        (
            mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
            mp_holistic.PoseLandmark.LEFT_HIP.value,
            mp_holistic.PoseLandmark.LEFT_KNEE.value,
            "L.Hip",
        ),
        (
            mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
            mp_holistic.PoseLandmark.RIGHT_HIP.value,
            mp_holistic.PoseLandmark.RIGHT_KNEE.value,
            "R.Hip",
        ),
    ]

    for a_idx, b_idx, c_idx, name in angle_definitions:
        if (
            a_idx in keypoint_positions
            and b_idx in keypoint_positions
            and c_idx in keypoint_positions
        ):
            a = keypoint_positions[a_idx][:2]
            b = keypoint_positions[b_idx][:2]
            c = keypoint_positions[c_idx][:2]

            vis_a = keypoint_positions[a_idx][2]
            vis_b = keypoint_positions[b_idx][2]
            vis_c = keypoint_positions[c_idx][2]

            if vis_a > 0.5 and vis_b > 0.5 and vis_c > 0.5:
                angle = angle_between_points(a, b, c)
                if angle is not None:
                    angles[name] = angle

    return angles
