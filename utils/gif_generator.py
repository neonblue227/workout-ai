"""
GIF generator from JSON keypoint data.

Creates animated GIFs with body, hands, and face visualizations
in a 2x2 grid layout from recorded keypoint data.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# MediaPipe pose connections (subset for visualization)
POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),  # Left face
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),  # Right face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13),
    (13, 15),  # Left arm
    (12, 14),
    (14, 16),  # Right arm
    (11, 23),
    (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25),
    (25, 27),  # Left leg
    (24, 26),
    (26, 28),  # Right leg
    (15, 17),
    (15, 19),
    (15, 21),  # Left hand
    (16, 18),
    (16, 20),
    (16, 22),  # Right hand
    (27, 29),
    (27, 31),  # Left foot
    (28, 30),
    (28, 32),  # Right foot
]

# Hand connections
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (5, 9),
    (9, 13),
    (13, 17),  # Palm
]

# Face mesh key contour connections (simplified)
FACE_CONTOUR_INDICES = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


def load_keypoints_json(filepath: str) -> dict:
    """
    Load keypoint data from JSON file.

    Args:
        filepath: Path to the JSON keypoint file

    Returns:
        dict: Parsed JSON data with metadata and frames
    """
    with open(filepath, "r") as f:
        return json.load(f)


def draw_body_panel(
    keypoints: Optional[dict],
    size: tuple[int, int],
    original_resolution: tuple[int, int],
) -> np.ndarray:
    """
    Draw body/pose landmarks on a panel.

    Args:
        keypoints: Pose keypoint data {idx: {x, y, z, visibility}}
        size: Panel size (width, height)
        original_resolution: Original video resolution for scaling

    Returns:
        np.ndarray: BGR image with pose landmarks drawn
    """
    panel = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark background

    # Draw title
    cv2.putText(
        panel, "BODY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    if keypoints is None:
        cv2.putText(
            panel,
            "No data",
            (size[0] // 2 - 40, size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            1,
        )
        return panel

    orig_w, orig_h = original_resolution
    scale_x = size[0] / orig_w
    scale_y = size[1] / orig_h

    # Collect scaled points
    points = {}
    for idx_str, kp in keypoints.items():
        idx = int(idx_str)
        x = int(kp["x"] * scale_x)
        y = int(kp["y"] * scale_y)
        visibility = kp.get("visibility", 1.0)
        points[idx] = (x, y, visibility)

    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            start = points[start_idx]
            end = points[end_idx]
            if start[2] > 0.3 and end[2] > 0.3:
                cv2.line(
                    panel,
                    (start[0], start[1]),
                    (end[0], end[1]),
                    (100, 100, 100),
                    2,
                    cv2.LINE_AA,
                )

    # Draw points
    for idx, (x, y, vis) in points.items():
        if vis < 0.1:
            continue
        # Color based on visibility
        if vis > 0.7:
            color = (0, 255, 0)  # Green - high confidence
        elif vis > 0.4:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 0, 255)  # Red - low

        cv2.circle(panel, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return panel


def draw_hand_panel(
    keypoints: Optional[dict],
    label: str,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Draw hand landmarks on a panel.

    Args:
        keypoints: Hand keypoint data {idx: {x, y, z}}
        label: "Left Hand" or "Right Hand"
        size: Panel size (width, height)

    Returns:
        np.ndarray: BGR image with hand landmarks drawn
    """
    panel = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    # Draw title with color coding
    title_color = (255, 150, 50) if "Left" in label else (50, 255, 150)
    cv2.putText(
        panel, label.upper(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2
    )

    if keypoints is None:
        cv2.putText(
            panel,
            "No data",
            (size[0] // 2 - 40, size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            1,
        )
        return panel

    # Find bounding box of hand points to center and scale
    xs = [kp["x"] for kp in keypoints.values()]
    ys = [kp["y"] for kp in keypoints.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    hand_w = max_x - min_x
    hand_h = max_y - min_y

    if hand_w < 1 or hand_h < 1:
        return panel

    # Calculate scale to fit panel with padding
    padding = 40
    available_w = size[0] - 2 * padding
    available_h = size[1] - 2 * padding - 30  # Account for title

    scale = min(available_w / hand_w, available_h / hand_h)

    # Center offset
    center_x = padding + (available_w - hand_w * scale) / 2
    center_y = padding + 30 + (available_h - hand_h * scale) / 2

    # Collect scaled points
    points = {}
    for idx_str, kp in keypoints.items():
        idx = int(idx_str)
        x = int((kp["x"] - min_x) * scale + center_x)
        y = int((kp["y"] - min_y) * scale + center_y)
        points[idx] = (x, y)

    # Draw connections
    line_color = (200, 120, 40) if "Left" in label else (40, 200, 120)
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(
                panel, points[start_idx], points[end_idx], line_color, 2, cv2.LINE_AA
            )

    # Draw points
    for idx, (x, y) in points.items():
        # Fingertips larger
        if idx in [4, 8, 12, 16, 20]:
            cv2.circle(panel, (x, y), 6, title_color, -1, cv2.LINE_AA)
            cv2.circle(panel, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(panel, (x, y), 4, line_color, -1, cv2.LINE_AA)

    return panel


def draw_face_panel(
    keypoints: Optional[dict],
    size: tuple[int, int],
) -> np.ndarray:
    """
    Draw face mesh landmarks on a panel.

    Args:
        keypoints: Face keypoint data {idx: {x, y, z}}
        size: Panel size (width, height)

    Returns:
        np.ndarray: BGR image with face landmarks drawn
    """
    panel = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    # Draw title
    cv2.putText(
        panel, "FACE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2
    )

    if keypoints is None:
        cv2.putText(
            panel,
            "No data",
            (size[0] // 2 - 40, size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            1,
        )
        return panel

    # Find bounding box to center and scale
    xs = [kp["x"] for kp in keypoints.values()]
    ys = [kp["y"] for kp in keypoints.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    face_w = max_x - min_x
    face_h = max_y - min_y

    if face_w < 1 or face_h < 1:
        return panel

    # Calculate scale
    padding = 30
    available_w = size[0] - 2 * padding
    available_h = size[1] - 2 * padding - 30

    scale = min(available_w / face_w, available_h / face_h)

    center_x = padding + (available_w - face_w * scale) / 2
    center_y = padding + 30 + (available_h - face_h * scale) / 2

    # Collect scaled points
    points = {}
    for idx_str, kp in keypoints.items():
        idx = int(idx_str)
        x = int((kp["x"] - min_x) * scale + center_x)
        y = int((kp["y"] - min_y) * scale + center_y)
        points[idx] = (x, y)

    # Draw face contour
    contour_pts = []
    for idx in FACE_CONTOUR_INDICES:
        if idx in points:
            contour_pts.append(points[idx])

    if len(contour_pts) > 2:
        pts = np.array(contour_pts, dtype=np.int32)
        cv2.polylines(panel, [pts], True, (100, 150, 200), 2, cv2.LINE_AA)

    # Draw key facial landmarks
    key_indices = {
        1: (255, 200, 100),  # Nose tip
        33: (255, 255, 0),  # Left eye outer
        263: (255, 255, 0),  # Right eye outer
        61: (200, 100, 255),  # Mouth left
        291: (200, 100, 255),  # Mouth right
        199: (100, 200, 255),  # Chin
    }

    for idx, color in key_indices.items():
        if idx in points:
            cv2.circle(panel, points[idx], 5, color, -1, cv2.LINE_AA)

    # Draw sparse mesh points
    for idx in range(0, len(points), 10):  # Every 10th point
        if idx in points and idx not in key_indices:
            cv2.circle(panel, points[idx], 2, (80, 80, 80), -1, cv2.LINE_AA)

    return panel


def create_frame_grid(
    pose: Optional[dict],
    left_hand: Optional[dict],
    right_hand: Optional[dict],
    face: Optional[dict],
    panel_size: tuple[int, int],
    original_resolution: tuple[int, int],
) -> np.ndarray:
    """
    Combine 4 panels into a 2x2 grid.

    Layout:
    +------------+------------+
    |   BODY     |   FACE     |
    +------------+------------+
    | LEFT HAND  | RIGHT HAND |
    +------------+------------+

    Args:
        pose: Pose keypoint data
        left_hand: Left hand keypoint data
        right_hand: Right hand keypoint data
        face: Face keypoint data
        panel_size: Size of each panel (width, height)
        original_resolution: Original video resolution

    Returns:
        np.ndarray: Combined BGR image
    """
    # Create individual panels
    body_panel = draw_body_panel(pose, panel_size, original_resolution)
    face_panel = draw_face_panel(face, panel_size)
    left_panel = draw_hand_panel(left_hand, "Left Hand", panel_size)
    right_panel = draw_hand_panel(right_hand, "Right Hand", panel_size)

    # Combine into grid
    top_row = np.hstack([body_panel, face_panel])
    bottom_row = np.hstack([left_panel, right_panel])
    grid = np.vstack([top_row, bottom_row])

    # Add border lines
    h, w = grid.shape[:2]
    cv2.line(grid, (w // 2, 0), (w // 2, h), (60, 60, 60), 2)
    cv2.line(grid, (0, h // 2), (w, h // 2), (60, 60, 60), 2)

    return grid


def generate_gif(
    json_path: str,
    output_path: Optional[str] = None,
    fps: Optional[float] = None,
    panel_size: tuple[int, int] = (320, 240),
) -> str:
    """
    Generate animated GIF from JSON keypoint data.

    Args:
        json_path: Path to the JSON keypoint file
        output_path: Output GIF path (default: data/gif/{filename}.gif)
        fps: Override FPS (default: use metadata FPS)
        panel_size: Size of each panel in the 2x2 grid

    Returns:
        str: Path to the generated GIF
    """
    # Load data
    data = load_keypoints_json(json_path)
    metadata = data["metadata"]
    frames_data = data["frames"]

    # Get metadata
    original_resolution = (
        metadata["resolution"]["width"],
        metadata["resolution"]["height"],
    )
    recording_fps = fps or metadata.get("fps", 10.0)

    # Calculate frame duration in milliseconds
    frame_duration = int(1000 / recording_fps)

    # Determine output path
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(json_path))
        gif_dir = os.path.join(base_dir, "gif")
        os.makedirs(gif_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(gif_dir, f"{filename}.gif")

    print(f"Generating GIF from {len(frames_data)} frames...")
    print(f"Original resolution: {original_resolution}")
    print(f"Panel size: {panel_size}")
    print(f"Output: {output_path}")

    # Generate frames
    pil_frames = []
    for i, frame in enumerate(frames_data):
        # Create grid frame
        grid = create_frame_grid(
            pose=frame.get("pose"),
            left_hand=frame.get("left_hand"),
            right_hand=frame.get("right_hand"),
            face=frame.get("face"),
            panel_size=panel_size,
            original_resolution=original_resolution,
        )

        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        pil_frames.append(pil_frame)

        # Progress
        if (i + 1) % 50 == 0 or i == len(frames_data) - 1:
            print(f"  Processed {i + 1}/{len(frames_data)} frames")

    # Save as GIF
    print("Saving GIF...")
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration,
        loop=0,  # Loop forever
        optimize=True,
    )

    print(f"GIF saved: {output_path}")

    return output_path
