"""
FPS calibration utility.

Measures actual processing FPS by running test frames through the pipeline.
"""

import sys
import time

import cv2


def calibrate_fps(cap, holistic, num_frames=30):
    """
    Run a calibration phase to measure actual processing FPS.

    Processes test frames to measure real processing time and returns
    the measured FPS based on actual performance.

    Args:
        cap: OpenCV VideoCapture object
        holistic: MediaPipe Holistic model instance
        num_frames: Number of frames to process for calibration (default: 30)

    Returns:
        float: Measured FPS based on real processing time
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
