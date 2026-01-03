"""
Face mesh drawing utility.

Draws 478 face mesh landmarks with tesselation, contours, and iris tracking.
"""

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def draw_face_mesh(
    frame, landmarks, draw_tesselation=True, draw_contours=True, draw_irises=True
):
    """
    Draw high-precision face mesh with 478 landmarks.

    Includes face contours, tesselation, and iris tracking.

    Args:
        frame: OpenCV image/frame to draw on
        landmarks: MediaPipe face landmarks object
        draw_tesselation: Whether to draw face mesh triangles
        draw_contours: Whether to draw face contours
        draw_irises: Whether to draw iris tracking

    Returns:
        int: Number of face points drawn
    """
    h, w = frame.shape[:2]
    face_points = []

    # Collect all face points
    for landmark in landmarks.landmark:
        px = int(round(landmark.x * w))
        py = int(round(landmark.y * h))
        face_points.append((px, py))

    # Draw tesselation (face mesh triangles) - subtle gray
    if draw_tesselation:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 80, 80), thickness=1
            ),
        )

    # Draw face contours - more visible
    if draw_contours:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

    # Draw irises - cyan color for visibility
    if draw_irises:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 0), thickness=2
            ),
        )

    # Draw key facial landmarks with dots
    # Key points: nose tip (1), left eye outer (33), right eye outer (263),
    # mouth left (61), mouth right (291), chin (199)
    key_face_indices = [1, 33, 263, 61, 291, 199, 4, 5, 6]
    for idx in key_face_indices:
        if idx < len(face_points):
            px, py = face_points[idx]
            cv2.circle(
                frame,
                (px, py),
                radius=3,
                color=(255, 200, 100),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    return len(face_points)
