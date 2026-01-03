"""
Keypoint recording utility.

Manages frame-by-frame keypoint collection and JSON serialization
for saving alongside video recordings.
"""

import json
import os
from datetime import datetime
from typing import Any


class KeypointRecorder:
    """
    Records keypoints frame-by-frame and saves to JSON.

    Collects keypoint data for each frame during video recording
    and serializes to a structured JSON file.

    Attributes:
        base_filename: Base name for the output file (without extension)
        resolution: Video resolution as (width, height) tuple
        fps: Recording frames per second
        frames: List of frame data dictionaries
    """

    def __init__(
        self,
        base_filename: str,
        resolution: tuple[int, int],
        fps: float,
    ):
        """
        Initialize the keypoint recorder.

        Args:
            base_filename: Base name for output file (e.g., "holistic_record_1")
            resolution: Video resolution as (width, height) tuple
            fps: Recording frames per second
        """
        self.base_filename = base_filename
        self.resolution = resolution
        self.fps = fps
        self.frames: list[dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()

    def add_frame(
        self,
        frame_number: int,
        timestamp: float,
        keypoints: dict,
    ) -> None:
        """
        Add keypoint data for a single frame.

        Args:
            frame_number: Sequential frame number (0-indexed)
            timestamp: Elapsed time in seconds since recording started
            keypoints: Keypoint data from extract_all_keypoints()
        """
        frame_data = {
            "frame": frame_number,
            "timestamp": round(timestamp, 4),
            **keypoints,
        }
        self.frames.append(frame_data)

    def save(self, output_folder: str) -> str:
        """
        Save collected keypoints to JSON file.

        Creates the output directory if it doesn't exist.
        The filename matches the video filename with .json extension.

        Args:
            output_folder: Directory path to save the JSON file

        Returns:
            str: Full path to the saved JSON file
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_data = {
            "metadata": {
                "filename": self.base_filename,
                "resolution": {
                    "width": self.resolution[0],
                    "height": self.resolution[1],
                },
                "fps": round(self.fps, 2),
                "total_frames": len(self.frames),
                "created_at": self.created_at,
            },
            "frames": self.frames,
        }

        output_path = os.path.join(output_folder, f"{self.base_filename}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return output_path

    def get_frame_count(self) -> int:
        """
        Get the number of frames recorded.

        Returns:
            int: Number of frames recorded so far
        """
        return len(self.frames)

    def get_detection_summary(self) -> dict[str, int]:
        """
        Get summary of detection counts.

        Returns:
            dict: Count of frames with each body part detected
        """
        summary = {
            "pose_detected": 0,
            "face_detected": 0,
            "left_hand_detected": 0,
            "right_hand_detected": 0,
        }

        for frame in self.frames:
            if frame.get("pose") is not None:
                summary["pose_detected"] += 1
            if frame.get("face") is not None:
                summary["face_detected"] += 1
            if frame.get("left_hand") is not None:
                summary["left_hand_detected"] += 1
            if frame.get("right_hand") is not None:
                summary["right_hand_detected"] += 1

        return summary
