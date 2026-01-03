"""
High-precision video recording with MediaPipe Holistic keypoint overlay.

Features:
- Full camera resolution capture
- MediaPipe Holistic for comprehensive detection:
  - 33 pose landmarks
  - 478 face landmarks (with iris tracking via refine_face_landmarks)
  - 21 landmarks per hand (42 total for both hands)
- model_complexity=2 for highest precision
- Sub-pixel precision keypoint rendering
- Visibility-based keypoint coloring
- Real-time joint angle display
- Saves original resolution video to data/video/
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
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


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


def draw_pose_landmarks(frame, landmarks, connections):
    """Draw pose landmarks with high precision and visibility-based coloring."""
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


def draw_face_mesh(
    frame, landmarks, draw_tesselation=True, draw_contours=True, draw_irises=True
):
    """
    Draw high-precision face mesh with 478 landmarks.
    Includes face contours, tesselation, and iris tracking.
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


def draw_hand_landmarks(frame, landmarks, hand_label, connections):
    """
    Draw high-precision hand landmarks (21 per hand).
    Color-coded: Left hand = blue tones, Right hand = green tones
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


def calculate_joint_angles(keypoint_positions):
    """Calculate important joint angles for pose analysis."""
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


def draw_info_overlay(frame, elapsed, duration, fps, detection_status):
    """Draw recording status and detection info on frame."""
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


def calibrate_fps(cap, holistic, num_frames=30):
    """
    Run a calibration phase to measure actual processing FPS.
    Returns the measured FPS based on real processing time.
    """
    print("\nCalibrating FPS (processing test frames)...")
    frame_times = []

    for i in range(num_frames):
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        # Process with MediaPipe (same as recording)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        holistic.process(rgb_frame)

        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        # Show progress
        sys.stdout.write(f"\r  Calibrating: {i + 1}/{num_frames} frames")
        sys.stdout.flush()

    print()

    if frame_times:
        avg_frame_time = sum(frame_times) / len(frame_times)
        measured_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 15.0
        return measured_fps
    return 15.0  # Default fallback


def record_with_keypoints():
    """Main recording function with high-precision Holistic keypoint overlay."""
    RECORD_FOLDER = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "video")
    )

    print("=" * 60)
    print("  High-Precision Holistic Keypoint Video Recorder")
    print("  Pose (33) + Face (478) + Hands (21 each) = 553 keypoints")
    print("=" * 60)

    # Get filename prefix
    base_name = input("\nFilename prefix (default: 'holistic_record'): ").strip()
    if not base_name:
        base_name = "holistic_record"

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

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Camera reported FPS: {camera_fps}")

    # Initialize MediaPipe Holistic with highest precision
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,  # Highest precision
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,  # Enable 478 face landmarks with iris
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Calibrate to get actual processing FPS
    measured_fps = calibrate_fps(cap, holistic, num_frames=30)
    print(f"Measured processing FPS: {measured_fps:.1f}")

    # Setup video writer with MEASURED FPS (not camera reported FPS)
    output_path = get_next_filename(RECORD_FOLDER, base_name, ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, measured_fps, (actual_width, actual_height)
    )

    print(f"\nOutput: {output_path}")
    print(f"Duration: {duration}s")
    print(f"Recording FPS: {measured_fps:.1f} (actual processing speed)")
    print("\nDetection includes:")
    print("  - Pose: 33 body landmarks")
    print("  - Face: 478 landmarks (with iris tracking)")
    print("  - Hands: 21 landmarks per hand")
    print("\nStarting recording... Press 'q' to stop early.")
    print("-" * 60)

    # Main recording loop
    with holistic:
        start_time = time.time()
        frame_count = 0
        fps_calc = measured_fps  # Start with calibrated value

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            ret, frame = cap.read()
            if not ret:
                print("\nError: Cannot read frame.")
                break

            frame_start = time.time()

            # Process with MediaPipe Holistic
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Create overlay frame
            overlay_frame = frame.copy()

            # Detection status tracking
            detection_status = {
                "pose": False,
                "pose_points": 0,
                "face": False,
                "face_points": 0,
                "left_hand": False,
                "left_hand_points": 0,
                "right_hand": False,
                "right_hand_points": 0,
            }

            keypoint_positions = {}

            # Draw pose landmarks
            if results.pose_landmarks:
                detection_status["pose"] = True
                detection_status["pose_points"] = 33
                keypoint_positions = draw_pose_landmarks(
                    overlay_frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                )

                # Calculate and draw angles
                angles = calculate_joint_angles(keypoint_positions)
                draw_angle_overlay(overlay_frame, angles)

            # Draw face mesh
            if results.face_landmarks:
                detection_status["face"] = True
                face_count = draw_face_mesh(
                    overlay_frame,
                    results.face_landmarks,
                    draw_tesselation=True,
                    draw_contours=True,
                    draw_irises=True,
                )
                detection_status["face_points"] = face_count

            # Draw left hand
            if results.left_hand_landmarks:
                detection_status["left_hand"] = True
                hand_count = draw_hand_landmarks(
                    overlay_frame,
                    results.left_hand_landmarks,
                    "Left",
                    mp_hands.HAND_CONNECTIONS,
                )
                detection_status["left_hand_points"] = hand_count

            # Draw right hand
            if results.right_hand_landmarks:
                detection_status["right_hand"] = True
                hand_count = draw_hand_landmarks(
                    overlay_frame,
                    results.right_hand_landmarks,
                    "Right",
                    mp_hands.HAND_CONNECTIONS,
                )
                detection_status["right_hand_points"] = hand_count

            # Draw info overlay
            draw_info_overlay(
                overlay_frame, elapsed, duration, fps_calc, detection_status
            )

            # Write frame to video
            out.write(overlay_frame)

            # Display preview (scaled down if needed)
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

            cv2.imshow("Holistic Recording (press 'q' to stop)", display_frame)

            # Calculate FPS
            frame_time = time.time() - frame_start
            fps_calc = 1.0 / frame_time if frame_time > 0 else 0

            # Progress bar
            percent = min(100, int((elapsed / duration) * 100))
            bar_len = 40
            filled = int(bar_len * percent // 100)
            bar = "=" * filled + "-" * (bar_len - filled)

            total_pts = (
                detection_status["pose_points"]
                + detection_status["face_points"]
                + detection_status["left_hand_points"]
                + detection_status["right_hand_points"]
            )
            sys.stdout.write(
                f"\r[{bar}] {percent}% | {elapsed:.1f}s | {total_pts} keypoints"
            )
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

    print("=" * 60)
    print("  Recording Complete!")
    print("=" * 60)
    print(f"Frames recorded: {frame_count}")
    print(f"File saved: {output_path}")
    print(f"Resolution: {actual_width}x{actual_height}")


if __name__ == "__main__":
    record_with_keypoints()
