"""
Hand landmark drawing utility.

Draws 21 hand landmarks per hand with color-coding for left and right hands.
"""

import cv2


def draw_hand_landmarks(frame, landmarks, hand_label, connections):
    """
    Draw high-precision hand landmarks (21 per hand).

    Color-coded: Left hand = blue tones, Right hand = green tones

    Args:
        frame: OpenCV image/frame to draw on
        landmarks: MediaPipe hand landmarks object
        hand_label: "Left" or "Right" to determine color scheme
        connections: Hand connections (e.g., mp_hands.HAND_CONNECTIONS)

    Returns:
        int: Number of hand points drawn
    """
    h, w = frame.shape[:2]
    hand_points = []

    # Color based on hand
    if hand_label == "Left":
        base_color = (255, 150, 50)  # Blue-ish for left
        joint_color = (255, 200, 100)
    else:
        base_color = (50, 255, 150)  # Green-ish for right
        joint_color = (100, 255, 200)

    for idx, landmark in enumerate(landmarks.landmark):
        px = int(round(landmark.x * w))
        py = int(round(landmark.y * h))
        hand_points.append((px, py))

        # Fingertips (indices 4, 8, 12, 16, 20) get larger circles
        if idx in [4, 8, 12, 16, 20]:
            cv2.circle(
                frame,
                (px, py),
                radius=6,
                color=joint_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (px, py),
                radius=7,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        # Knuckles and joints
        elif idx in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
            cv2.circle(
                frame,
                (px, py),
                radius=4,
                color=base_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # Draw hand connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(hand_points) and end_idx < len(hand_points):
                cv2.line(
                    frame,
                    hand_points[start_idx],
                    hand_points[end_idx],
                    color=base_color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

    return len(hand_points)
