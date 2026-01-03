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
- Saves original resolution video to data/record/
"""

import os
import sys
import time

import cv2
import mediapipe as mp

# Add project root to sys.path to allow importing from utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import (
    calculate_joint_angles,
    calibrate_fps,
    draw_angle_overlay,
    draw_face_mesh,
    draw_hand_landmarks,
    draw_info_overlay,
    draw_pose_landmarks,
    get_next_filename,
)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands


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
