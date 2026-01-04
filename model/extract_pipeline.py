"""
Extract Pipeline for Training Data.

Processes raw MP4 videos to extract MediaPipe Holistic keypoints
and generate training data artifacts:
- Video with keypoint overlay (.mp4)
- Keypoint data in JSON format (.json)
- Visualization GIF (.gif)
"""

import glob
import os
import sys
import time

import cv2
import mediapipe as mp

# Path setup for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils import (
        KeypointRecorder,
        draw_face_mesh,
        draw_hand_landmarks,
        draw_pose_landmarks,
        extract_all_keypoints,
        generate_gif,
    )
except ImportError:
    print("Please run 'pip install -r requirements.txt' to install dependencies.")
    sys.exit(1)


# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands


def get_video_files(raw_folder: str) -> list[str]:
    """
    Get all video files from the raw folder.

    Args:
        raw_folder: Path to the folder containing raw videos

    Returns:
        List of absolute paths to video files
    """
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(raw_folder, ext)))
        # Also check uppercase extensions
        video_files.extend(glob.glob(os.path.join(raw_folder, ext.upper())))

    return sorted(set(video_files))


def process_video(
    video_path: str,
    overlay_folder: str,
    keypoint_folder: str,
    gif_folder: str,
) -> dict:
    """
    Process a single video file and generate all outputs.

    Args:
        video_path: Path to the input video file
        overlay_folder: Output folder for overlay videos
        keypoint_folder: Output folder for JSON keypoints
        gif_folder: Output folder for GIFs

    Returns:
        dict: Paths to generated files
    """
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"\n{'=' * 60}")
    print(f"Processing: {base_name}")
    print(f"{'=' * 60}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video: {video_path}")
        return {}

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Create output directories
    os.makedirs(overlay_folder, exist_ok=True)
    os.makedirs(keypoint_folder, exist_ok=True)
    os.makedirs(gif_folder, exist_ok=True)

    # Setup video writer for overlay output
    overlay_path = os.path.join(overlay_folder, f"{base_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    # Setup keypoint recorder
    recorder = KeypointRecorder(
        base_filename=base_name,
        resolution=(width, height),
        fps=fps,
    )

    # Process with MediaPipe Holistic
    frame_count = 0
    start_time = time.time()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process with MediaPipe
            results = holistic.process(rgb_frame)

            # Make frame writeable again
            rgb_frame.flags.writeable = True

            # Draw landmarks on frame
            overlay_frame = frame.copy()

            # Draw pose landmarks
            if results.pose_landmarks:
                draw_pose_landmarks(
                    overlay_frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                )

            # Draw face mesh
            if results.face_landmarks:
                draw_face_mesh(
                    overlay_frame,
                    results.face_landmarks,
                    draw_tesselation=True,
                    draw_contours=True,
                    draw_irises=True,
                )

            # Draw left hand
            if results.left_hand_landmarks:
                draw_hand_landmarks(
                    overlay_frame,
                    results.left_hand_landmarks,
                    "Left",
                    mp_hands.HAND_CONNECTIONS,
                )

            # Draw right hand
            if results.right_hand_landmarks:
                draw_hand_landmarks(
                    overlay_frame,
                    results.right_hand_landmarks,
                    "Right",
                    mp_hands.HAND_CONNECTIONS,
                )

            # Write overlay frame
            out.write(overlay_frame)

            # Extract and record keypoints
            keypoints = extract_all_keypoints(results, frame.shape)
            timestamp = frame_count / fps
            recorder.add_frame(frame_count, timestamp, keypoints)

            frame_count += 1

            # Progress update
            if frame_count % 30 == 0 or frame_count == total_frames:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(
                    f"  Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_actual:.1f} fps"
                )

    # Cleanup video resources
    cap.release()
    out.release()

    # Save keypoints to JSON
    json_path = recorder.save(keypoint_folder)

    # Get detection summary
    summary = recorder.get_detection_summary()
    print("\n  Detection Summary:")
    print(f"    Pose: {summary['pose_detected']}/{frame_count} frames")
    print(f"    Face: {summary['face_detected']}/{frame_count} frames")
    print(f"    Left Hand: {summary['left_hand_detected']}/{frame_count} frames")
    print(f"    Right Hand: {summary['right_hand_detected']}/{frame_count} frames")

    # Generate GIF from keypoints
    print("\n  Generating GIF...")
    gif_path = os.path.join(gif_folder, f"{base_name}.gif")
    generate_gif(
        json_path=json_path,
        output_path=gif_path,
        fps=min(fps, 15),  # Cap GIF FPS to reduce file size
        panel_size=(320, 240),
    )

    elapsed_total = time.time() - start_time
    print(f"\n  Completed in {elapsed_total:.1f} seconds")
    print("  Outputs:")
    print(f"    Overlay: {overlay_path}")
    print(f"    Keypoints: {json_path}")
    print(f"    GIF: {gif_path}")

    return {
        "overlay": overlay_path,
        "keypoints": json_path,
        "gif": gif_path,
    }


def main():
    """Main entry point for the extraction pipeline."""
    print("=" * 60)
    print("  MoveUp-AI: Training Data Extraction Pipeline")
    print("  Extracts keypoints from raw videos for model training")
    print("=" * 60)

    # Default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "neck_stretch")

    raw_folder = os.path.join(data_dir, "raw")
    overlay_folder = os.path.join(data_dir, "overlay")
    keypoint_folder = os.path.join(data_dir, "keypoint")
    gif_folder = os.path.join(data_dir, "gif")

    # Allow command-line override for raw folder
    if len(sys.argv) > 1:
        raw_folder = os.path.abspath(sys.argv[1])
        data_dir = os.path.dirname(raw_folder)
        overlay_folder = os.path.join(data_dir, "overlay")
        keypoint_folder = os.path.join(data_dir, "keypoint")
        gif_folder = os.path.join(data_dir, "gif")

    print(f"\nInput folder: {raw_folder}")
    print("Output folders:")
    print(f"  Overlay: {overlay_folder}")
    print(f"  Keypoints: {keypoint_folder}")
    print(f"  GIF: {gif_folder}")

    # Check if raw folder exists
    if not os.path.exists(raw_folder):
        print(f"\nERROR: Raw folder not found: {raw_folder}")
        print("Please ensure the folder exists and contains video files.")
        sys.exit(1)

    # Get video files
    video_files = get_video_files(raw_folder)

    if not video_files:
        print(f"\nNo video files found in: {raw_folder}")
        print("Supported formats: mp4, mov, avi, mkv, webm")
        sys.exit(1)

    print(f"\nFound {len(video_files)} video(s):")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")

    # Process each video
    results = []
    for video_path in video_files:
        try:
            result = process_video(
                video_path=video_path,
                overlay_folder=overlay_folder,
                keypoint_folder=keypoint_folder,
                gif_folder=gif_folder,
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR processing {os.path.basename(video_path)}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\nProcessed: {len(results)}/{len(video_files)} videos")

    if results:
        print("\nGenerated files:")
        for r in results:
            print(f"\n  {os.path.basename(r['keypoints']).replace('.json', '')}:")
            print(f"    • Overlay video: {os.path.basename(r['overlay'])}")
            print(f"    • Keypoints JSON: {os.path.basename(r['keypoints'])}")
            print(f"    • Visualization GIF: {os.path.basename(r['gif'])}")


if __name__ == "__main__":
    main()
