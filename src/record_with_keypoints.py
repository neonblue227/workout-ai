"""
Video Recording Module with High-Precision Keypoint Extraction

Records video from webcam, extracts MediaPipe pose keypoints with maximum
precision, and saves MP4 with skeleton overlay to the record folder.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path


# MediaPipe setup with highest precision settings
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing specs for better visualization
LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(0, 255, 0),  # Green dots
    thickness=2,
    circle_radius=3
)
CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(255, 0, 0),  # Blue lines (BGR)
    thickness=2
)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_next_filename(output_dir: Path, prefix: str = "recording") -> tuple[str, int]:
    """
    Find the next available filename number.

    Returns:
        tuple: (filename without extension, number)
    """
    existing = list(output_dir.glob(f"{prefix}_*.mp4"))
    if not existing:
        return f"{prefix}_1", 1

    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[-1])
            numbers.append(num)
        except ValueError:
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}_{next_num}", next_num


def normalized_to_pixel(landmark, image_shape: tuple) -> tuple[int, int]:
    """
    Convert MediaPipe normalized coordinates to pixel coordinates.

    Args:
        landmark: MediaPipe landmark with x, y attributes (0-1 normalized)
        image_shape: (height, width, channels) of the image

    Returns:
        tuple: (x, y) pixel coordinates
    """
    height, width = image_shape[:2]
    return (int(landmark.x * width), int(landmark.y * height))


def keypoints_to_dict(landmarks, image_shape: tuple) -> dict:
    """
    Convert MediaPipe landmarks to dictionary with pixel coordinates.

    Args:
        landmarks: MediaPipe pose landmarks
        image_shape: (height, width, channels) of the image

    Returns:
        dict: Named keypoints with x, y, z, visibility values
    """
    keypoints = {}
    for idx, landmark in enumerate(landmarks.landmark):
        name = mp_pose.PoseLandmark(idx).name
        x_px, y_px = normalized_to_pixel(landmark, image_shape)
        keypoints[name] = {
            "x": x_px,
            "y": y_px,
            "z": landmark.z,  # Depth relative to hips
            "visibility": landmark.visibility,
            "x_normalized": landmark.x,
            "y_normalized": landmark.y
        }
    return keypoints


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Calculate angle at point b formed by points a, b, c.

    Args:
        a, b, c: (x, y) coordinates of three points

    Returns:
        float: Angle in degrees at point b
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return angle


def draw_angle_info(frame, keypoints: dict, angle_configs: list) -> None:
    """
    Draw angle measurements on frame.

    Args:
        frame: OpenCV frame to draw on
        keypoints: Dictionary of keypoints
        angle_configs: List of (point_a, point_b, point_c, label) tuples
    """
    for config in angle_configs:
        try:
            p_a, p_b, p_c, label = config
            a = (keypoints[p_a]["x"], keypoints[p_a]["y"])
            b = (keypoints[p_b]["x"], keypoints[p_b]["y"])
            c = (keypoints[p_c]["x"], keypoints[p_c]["y"])

            # Check visibility
            min_vis = min(
                keypoints[p_a]["visibility"],
                keypoints[p_b]["visibility"],
                keypoints[p_c]["visibility"]
            )

            if min_vis > 0.5:
                angle = calculate_angle(a, b, c)
                cv2.putText(
                    frame,
                    f"{label}: {angle:.1f}°",
                    (b[0] + 10, b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        except KeyError:
            continue


class VideoRecorderWithKeypoints:
    """
    Records video with real-time pose keypoint extraction and overlay.
    """

    def __init__(
        self,
        output_dir: str | Path = None,
        resolution: tuple[int, int] = (1280, 720),
        fps: float = 30.0,
        model_complexity: int = 2,  # 0, 1, or 2 (highest precision)
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        show_angles: bool = True
    ):
        """
        Initialize the video recorder.

        Args:
            output_dir: Directory to save recordings (default: project/record)
            resolution: (width, height) for video capture
            fps: Frames per second for output video
            model_complexity: MediaPipe model complexity (2 = highest precision)
            min_detection_confidence: Minimum detection confidence threshold
            min_tracking_confidence: Minimum tracking confidence threshold
            show_angles: Whether to display angle measurements
        """
        self.project_root = get_project_root()
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "record"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resolution = resolution
        self.fps = fps
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.show_angles = show_angles

        # Angle configurations: (point_a, point_b, point_c, label)
        self.angle_configs = [
            ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "L.Elbow"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "R.Elbow"),
            ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "L.Knee"),
            ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "R.Knee"),
            ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "L.Hip"),
            ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "R.Hip"),
        ]

        self.cap = None
        self.writer = None
        self.pose = None
        self.all_keypoints = []
        self.frame_count = 0

    def _setup_camera(self) -> bool:
        """Initialize camera with specified resolution."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual resolution (camera may not support requested)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        self.resolution = (actual_w, actual_h)

        return True

    def _setup_writer(self, filename: str) -> None:
        """Initialize video writer."""
        output_path = self.output_dir / f"{filename}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            self.resolution
        )
        print(f"Output: {output_path}")

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict | None]:
        """
        Process a single frame for pose detection.

        Returns:
            tuple: (annotated_frame, keypoints_dict or None)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Detect pose
        results = self.pose.process(rgb_frame)

        # Prepare output frame
        annotated = frame.copy()
        keypoints = None

        if results.pose_landmarks:
            # Draw pose landmarks with custom styling
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=LANDMARK_STYLE,
                connection_drawing_spec=CONNECTION_STYLE
            )

            # Extract keypoints
            keypoints = keypoints_to_dict(results.pose_landmarks, frame.shape)

            # Draw angle info if enabled
            if self.show_angles:
                draw_angle_info(annotated, keypoints, self.angle_configs)

        return annotated, keypoints

    def record(
        self,
        duration: float = None,
        filename_prefix: str = "recording",
        mirror: bool = True
    ) -> tuple[str, str]:
        """
        Start recording video with keypoint extraction.

        Args:
            duration: Recording duration in seconds (None = until 'q' pressed)
            filename_prefix: Prefix for output files
            mirror: Whether to mirror the video horizontally

        Returns:
            tuple: (video_path, keypoints_path)
        """
        if not self._setup_camera():
            return None, None

        filename, num = get_next_filename(self.output_dir, filename_prefix)
        self._setup_writer(filename)

        self.all_keypoints = []
        self.frame_count = 0

        total_frames = int(duration * self.fps) if duration else float('inf')

        print(f"\nRecording started: {filename}")
        print("Press 'q' to stop recording")
        print("-" * 40)

        with mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            enable_segmentation=False,
            smooth_landmarks=True
        ) as pose:
            self.pose = pose

            while self.cap.isOpened() and self.frame_count < total_frames:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                # Mirror frame if requested
                if mirror:
                    frame = cv2.flip(frame, 1)

                # Process frame
                annotated_frame, keypoints = self._process_frame(frame)

                # Store keypoints
                if keypoints:
                    self.all_keypoints.append({
                        "frame": self.frame_count,
                        "timestamp": self.frame_count / self.fps,
                        "keypoints": keypoints
                    })

                # Write frame
                self.writer.write(annotated_frame)

                # Display preview
                # Add recording indicator
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    annotated_frame,
                    f"REC | Frame: {self.frame_count}",
                    (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                if duration:
                    progress = (self.frame_count / total_frames) * 100
                    cv2.putText(
                        annotated_frame,
                        f"Progress: {progress:.1f}%",
                        (50, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )

                cv2.imshow("Recording - Press 'q' to stop", annotated_frame)

                self.frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nRecording stopped by user")
                    break

        # Cleanup
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

        # Save keypoints
        video_path = self.output_dir / f"{filename}.mp4"
        keypoints_path = self.output_dir / f"{filename}_keypoints.json"

        with open(keypoints_path, 'w') as f:
            json.dump({
                "metadata": {
                    "filename": filename,
                    "resolution": list(self.resolution),
                    "fps": self.fps,
                    "total_frames": self.frame_count,
                    "duration_seconds": self.frame_count / self.fps,
                    "model_complexity": self.model_complexity,
                    "created_at": datetime.now().isoformat()
                },
                "frames": self.all_keypoints
            }, f, indent=2)

        print("-" * 40)
        print(f"Recording complete!")
        print(f"  Video: {video_path}")
        print(f"  Keypoints: {keypoints_path}")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Duration: {self.frame_count / self.fps:.2f}s")
        print(f"  Frames with keypoints: {len(self.all_keypoints)}")

        return str(video_path), str(keypoints_path)


def process_existing_video(
    input_path: str | Path,
    output_dir: str | Path = None,
    model_complexity: int = 2
) -> tuple[str, str]:
    """
    Process an existing video file and add keypoint overlay.

    Args:
        input_path: Path to input video
        output_dir: Output directory (default: project/record)
        model_complexity: MediaPipe model complexity (2 = highest)

    Returns:
        tuple: (output_video_path, keypoints_json_path)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    project_root = get_project_root()
    output_dir = Path(output_dir) if output_dir else project_root / "record"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing: {input_path.name}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")

    # Setup output
    output_name = f"{input_path.stem}_keypoints"
    output_video = output_dir / f"{output_name}.mp4"
    output_json = output_dir / f"{output_name}.json"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    all_keypoints = []
    frame_idx = 0

    with mp_pose.Pose(
        model_complexity=model_complexity,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        smooth_landmarks=True
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            annotated = frame.copy()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=LANDMARK_STYLE,
                    connection_drawing_spec=CONNECTION_STYLE
                )

                keypoints = keypoints_to_dict(results.pose_landmarks, frame.shape)
                all_keypoints.append({
                    "frame": frame_idx,
                    "timestamp": frame_idx / fps,
                    "keypoints": keypoints
                })

            writer.write(annotated)
            frame_idx += 1

            # Progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"\rProgress: {progress:.1f}%", end="", flush=True)

    cap.release()
    writer.release()

    # Save keypoints
    with open(output_json, 'w') as f:
        json.dump({
            "metadata": {
                "source": str(input_path),
                "output": str(output_video),
                "resolution": [width, height],
                "fps": fps,
                "total_frames": frame_idx,
                "model_complexity": model_complexity,
                "created_at": datetime.now().isoformat()
            },
            "frames": all_keypoints
        }, f, indent=2)

    print(f"\n\nProcessing complete!")
    print(f"  Output video: {output_video}")
    print(f"  Keypoints: {output_json}")
    print(f"  Frames with keypoints: {len(all_keypoints)}/{frame_idx}")

    return str(output_video), str(output_json)


def main():
    """Interactive CLI for video recording with keypoints."""
    print("=" * 50)
    print("Video Recording with High-Precision Keypoints")
    print("=" * 50)

    print("\nOptions:")
    print("  1. Record from webcam")
    print("  2. Process existing video")
    print("  q. Quit")

    choice = input("\nSelect option: ").strip().lower()

    if choice == '1':
        # Recording options
        print("\nResolution options:")
        print("  1. 640x480 (SD)")
        print("  2. 1280x720 (HD) [default]")
        print("  3. 1920x1080 (Full HD)")

        res_choice = input("Select resolution [2]: ").strip() or "2"
        resolutions = {
            "1": (640, 480),
            "2": (1280, 720),
            "3": (1920, 1080)
        }
        resolution = resolutions.get(res_choice, (1280, 720))

        duration_input = input("Duration in seconds (empty = manual stop): ").strip()
        duration = float(duration_input) if duration_input else None

        prefix = input("Filename prefix [recording]: ").strip() or "recording"

        recorder = VideoRecorderWithKeypoints(
            resolution=resolution,
            model_complexity=2,  # Highest precision
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        recorder.record(duration=duration, filename_prefix=prefix)

    elif choice == '2':
        video_path = input("Enter video path: ").strip()
        if video_path:
            try:
                process_existing_video(video_path)
            except Exception as e:
                print(f"Error: {e}")

    elif choice == 'q':
        print("Goodbye!")
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
