"""
Pose landmark drawing utility.

Draws 33 pose landmarks with visibility-based coloring and skeleton connections.
"""

import cv2

from .visibility_color import get_visibility_color


def draw_pose_landmarks(frame, landmarks, connections):
    """
    Draw pose landmarks with high precision and visibility-based coloring.

    Args:
        frame: OpenCV image/frame to draw on
        landmarks: MediaPipe pose landmarks object
        connections: Pose connections (e.g., mp_holistic.POSE_CONNECTIONS)

    Returns:
        dict: Keypoint positions as {idx: (px, py, visibility)}
    """
    h, w = frame.shape[:2]
    keypoint_positions = {}

    for idx, landmark in enumerate(landmarks.landmark):
        px = landmark.x * w
        py = landmark.y * h
        visibility = getattr(landmark, "visibility", 1.0)

        keypoint_positions[idx] = (px, py, visibility)

        if visibility < 0.1:
            continue

        color = get_visibility_color(visibility)

        # Draw filled circle with anti-aliasing
        cv2.circle(
            frame,
            (int(round(px)), int(round(py))),
            radius=5,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            (int(round(px)), int(round(py))),
            radius=6,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx in keypoint_positions and end_idx in keypoint_positions:
                start = keypoint_positions[start_idx]
                end = keypoint_positions[end_idx]

                if start[2] < 0.3 or end[2] < 0.3:
                    continue

                cv2.line(
                    frame,
                    (int(round(start[0])), int(round(start[1]))),
                    (int(round(end[0])), int(round(end[1]))),
                    color=(200, 200, 200),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

    return keypoint_positions
