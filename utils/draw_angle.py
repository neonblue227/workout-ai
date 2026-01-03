"""
Angle overlay drawing utility.

Draws joint angle values as text overlay on frame.
"""

import cv2


def draw_angle_overlay(frame, angles):
    """
    Draw joint angles on the frame.

    Displays angle values with shadow effect for readability.

    Args:
        frame: OpenCV image/frame to draw on
        angles: Dict of {joint_name: angle_degrees}
    """
    y_offset = 30
    for name, angle in angles.items():
        text = f"{name}: {angle:.1f}"
        # Shadow
        cv2.putText(
            frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        # Text
        cv2.putText(
            frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 25
