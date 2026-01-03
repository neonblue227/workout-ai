"""
High-precision video recording with MediaPipe keypoint overlay.

Features:
- Full camera resolution capture
- model_complexity=2 for highest precision pose detection
- Sub-pixel precision keypoint rendering
- Visibility-based keypoint coloring (green=high, yellow=medium, red=low)
- Real-time joint angle display
- Saves original resolution video to data/record/
"""

import os
import sys
import time
from math import acos, degrees

import cv2
import mediapipe as mp

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Custom drawing specifications for high-precision overlay
LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(0, 255, 0), thickness=2, circle_radius=3
)
CONNECTION_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)


def normalized_to_pixel(coord, image_shape):
    """Convert normalized MediaPipe coords to pixel coordinates."""
    h, w = image_shape[:2]
    return coord.x * w, coord.y * h


def angle_between_points(a, b, c):
    """
    Calculate angle (degrees) at point b formed by a-b-c.
    a, b, c are (x, y) tuples.
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


def get_visibility_color(visibility):
    """
    Return BGR color based on landmark visibility.
    Green (high) -> Yellow (medium) -> Red (low)
    """
    if visibility > 0.8:
        return (0, 255, 0)  # Green - high confidence
    elif visibility > 0.5:
        return (0, 255, 255)  # Yellow - medium confidence
    else:
        return (0, 0, 255)  # Red - low confidence


def draw_high_precision_keypoints(frame, landmarks, connections=None):
    """
    Draw keypoints with sub-pixel precision and visibility-based coloring.
    """
    h, w = frame.shape[:2]
    keypoint_positions = {}

    # Draw each landmark with visibility-based color
    for idx, landmark in enumerate(landmarks.landmark):
        # Sub-pixel precision coordinates
        px = landmark.x * w
        py = landmark.y * h
        visibility = landmark.visibility

        keypoint_positions[idx] = (px, py, visibility)

        # Skip if visibility is too low
        if visibility < 0.1:
            continue

        color = get_visibility_color(visibility)

        # Draw filled circle with anti-aliasing (sub-pixel precision)
        cv2.circle(
            frame,
            (int(round(px)), int(round(py))),
            radius=5,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        # Draw outer ring
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

                # Skip if either point has low visibility
                if start[2] < 0.3 or end[2] < 0.3:
                    continue

                # Draw anti-aliased line
                cv2.line(
                    frame,
                    (int(round(start[0])), int(round(start[1]))),
                    (int(round(end[0])), int(round(end[1]))),
                    color=(200, 200, 200),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

    return keypoint_positions


def calculate_joint_angles(keypoint_positions, h, w):
    """Calculate important joint angles for pose analysis."""
    angles = {}

    # Define angle calculations: (point_a, vertex, point_c, name)
    angle_definitions = [
        (
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
            "L.Elbow",
        ),
        (
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            "R.Elbow",
        ),
        (
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            "L.Knee",
        ),
        (
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            "R.Knee",
        ),
        (
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            "L.Hip",
        ),
        (
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
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

            # Check visibility threshold
            vis_a = keypoint_positions[a_idx][2]
            vis_b = keypoint_positions[b_idx][2]
            vis_c = keypoint_positions[c_idx][2]

            if vis_a > 0.5 and vis_b > 0.5 and vis_c > 0.5:
                angle = angle_between_points(a, b, c)
                if angle is not None:
                    angles[name] = angle

    return angles


def draw_angle_overlay(frame, angles):
    """Draw joint angles on the frame."""
    y_offset = 30
    for name, angle in angles.items():
        text = f"{name}: {angle:.1f}"
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


def draw_recording_info(frame, elapsed, duration, fps):
    """Draw recording status info on frame."""
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
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (w - 100, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Resolution
    res_text = f"{w}x{h}"
    cv2.putText(
        frame,
        res_text,
        (w - 100, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def get_next_filename(folder, base_name, extension=".mp4"):
    """Generate next filename in sequence."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        return os.path.join(folder, f"{base_name}_1{extension}")

    files = os.listdir(folder)
    max_num = 0
    for file in files:
        if file.startswith(base_name) and file.endswith(extension):
            try:
                part = file.replace(base_name + "_", "").replace(extension, "")
                num = int(part)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    return os.path.join(folder, f"{base_name}_{max_num + 1}{extension}")


def record_with_keypoints():
    """Main recording function with high-precision keypoint overlay."""
    # Output folder
    RECORD_FOLDER = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "record")
    )

    print("=" * 50)
    print("  High-Precision Keypoint Video Recorder")
    print("=" * 50)

    # Get filename prefix
    base_name = input("\nFilename prefix (default: 'keypoint_record'): ").strip()
    if not base_name:
        base_name = "keypoint_record"

    # Get duration
    try:
        duration = float(
            input("Recording duration in seconds (default: 30): ").strip() or "30"
        )
    except ValueError:
        duration = 30.0

    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Request maximum resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Camera FPS: {actual_fps}")

    # Setup video writer
    output_path = get_next_filename(RECORD_FOLDER, base_name, ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, actual_fps, (actual_width, actual_height)
    )

    print(f"\nOutput: {output_path}")
    print(f"Duration: {duration}s")
    print("\nStarting recording... Press 'q' to stop early.")
    print("-" * 50)

    # Initialize MediaPipe Pose with highest precision
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Highest precision (0, 1, or 2)
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        start_time = time.time()
        frame_count = 0
        fps_calc = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            ret, frame = cap.read()
            if not ret:
                print("\nError: Cannot read frame.")
                break

            frame_start = time.time()

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Create overlay frame
            overlay_frame = frame.copy()

            if results.pose_landmarks:
                # Draw high-precision keypoints
                keypoint_positions = draw_high_precision_keypoints(
                    overlay_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                # Calculate and draw angles
                angles = calculate_joint_angles(
                    keypoint_positions, actual_height, actual_width
                )
                draw_angle_overlay(overlay_frame, angles)

            # Draw recording info
            draw_recording_info(overlay_frame, elapsed, duration, fps_calc)

            # Write frame to video
            out.write(overlay_frame)

            # Display preview (scaled down if needed for performance)
            display_frame = overlay_frame
            if actual_width > 1280:
                scale = 1280 / actual_width
                display_frame = cv2.resize(
                    overlay_frame,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow("Recording (press 'q' to stop)", display_frame)

            # Calculate FPS
            frame_time = time.time() - frame_start
            fps_calc = 1.0 / frame_time if frame_time > 0 else 0

            # Progress bar
            percent = min(100, int((elapsed / duration) * 100))
            bar_len = 40
            filled = int(bar_len * percent // 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(f"\r[{bar}] {percent}% | {elapsed:.1f}s")
            sys.stdout.flush()

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n\nRecording stopped by user.")
                break

    # Cleanup
    print("\n")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("=" * 50)
    print("  Recording Complete!")
    print("=" * 50)
    print(f"Frames recorded: {frame_count}")
    print(f"File saved: {output_path}")
    print(f"Resolution: {actual_width}x{actual_height}")


if __name__ == "__main__":
    record_with_keypoints()
