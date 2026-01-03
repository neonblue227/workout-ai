"""
MoveUp-AI Utility Functions.

This module provides reusable utility functions for pose detection,
visualization, and video recording.
"""

from .angle import angle_between_points
from .draw_angle import draw_angle_overlay
from .draw_face import draw_face_mesh
from .draw_hand import draw_hand_landmarks
from .draw_info import draw_info_overlay
from .draw_pose import draw_pose_landmarks
from .file_utils import get_next_filename
from .fps_calibration import calibrate_fps
from .gif_generator import generate_gif
from .joint_angles import calculate_joint_angles
from .keypoint_extractor import (
    extract_all_keypoints,
    extract_face_keypoints,
    extract_hand_keypoints,
    extract_pose_keypoints,
)
from .keypoint_recorder import KeypointRecorder
from .visibility_color import get_visibility_color

__all__ = [
    "angle_between_points",
    "get_visibility_color",
    "draw_pose_landmarks",
    "draw_face_mesh",
    "draw_hand_landmarks",
    "calculate_joint_angles",
    "draw_info_overlay",
    "draw_angle_overlay",
    "get_next_filename",
    "calibrate_fps",
    "extract_all_keypoints",
    "extract_pose_keypoints",
    "extract_face_keypoints",
    "extract_hand_keypoints",
    "KeypointRecorder",
    "generate_gif",
]
