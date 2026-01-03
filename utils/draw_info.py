"""
Info overlay drawing utility.

Draws recording status panel with time, FPS, resolution, and detection counts.
"""

import cv2


def draw_info_overlay(frame, elapsed, duration, fps, detection_status):
    """
    Draw recording status and detection info on frame.

    Displays:
        - Recording indicator (red dot)
        - Time progress
        - FPS counter
        - Resolution
        - Detection status panel (pose, face, hands)

    Args:
        frame: OpenCV image/frame to draw on
        elapsed: Elapsed time in seconds
        duration: Total duration in seconds
        fps: Current FPS
        detection_status: Dict with detection states and counts
    """
    h, w = frame.shape[:2]

    # Recording indicator (red dot)
    cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)

    # Time info
    time_text = f"{elapsed:.1f}s / {duration:.1f}s"
    cv2.putText(
        frame,
        time_text,
        (w - 150, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 100, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Resolution
    cv2.putText(
        frame,
        f"{w}x{h}",
        (w - 100, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Detection status panel (bottom left)
    panel_y = h - 100
    cv2.rectangle(frame, (5, panel_y - 5), (180, h - 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, panel_y - 5), (180, h - 5), (100, 100, 100), 1)

    # Pose status
    pose_color = (0, 255, 0) if detection_status["pose"] else (100, 100, 100)
    cv2.putText(
        frame,
        f"Pose: {detection_status['pose_points']} pts",
        (10, panel_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        pose_color,
        1,
        cv2.LINE_AA,
    )

    # Face status
    face_color = (0, 255, 0) if detection_status["face"] else (100, 100, 100)
    cv2.putText(
        frame,
        f"Face: {detection_status['face_points']} pts",
        (10, panel_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        face_color,
        1,
        cv2.LINE_AA,
    )

    # Left hand status
    lh_color = (255, 150, 50) if detection_status["left_hand"] else (100, 100, 100)
    cv2.putText(
        frame,
        f"L.Hand: {detection_status['left_hand_points']} pts",
        (10, panel_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        lh_color,
        1,
        cv2.LINE_AA,
    )

    # Right hand status
    rh_color = (50, 255, 150) if detection_status["right_hand"] else (100, 100, 100)
    cv2.putText(
        frame,
        f"R.Hand: {detection_status['right_hand_points']} pts",
        (10, panel_y + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        rh_color,
        1,
        cv2.LINE_AA,
    )
