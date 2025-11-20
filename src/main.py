import json
from math import acos, degrees

import cv2
import mediapipe as mp

# --- Helpers ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def normalized_to_pixel(coord, image_shape):
    h, w = image_shape[:2]
    return int(coord.x * w), int(coord.y * h)


def angle_between_points(a, b, c):
    """
    Return the angle (in degrees) at point b formed by points a-b-c.
    a, b, c are (x, y) tuples in pixel coordinates.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    # vectors BA and BC
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    # dot and norms
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    norm2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return None
    # clamp to avoid numeric errors
    cosang = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return degrees(acos(cosang))


def keypoints_to_dict(landmarks, image_shape):
    """Convert MediaPipe landmarks to a dict of named pixel coords."""
    names = mp_pose.PoseLandmark
    kp = {}
    for lm_name in names:
        idx = lm_name.value
        lm = landmarks[idx]
        x, y = normalized_to_pixel(lm, image_shape)
        kp[lm_name.name] = {
            "x": x,
            "y": y,
            "visibility": lm.visibility if hasattr(lm, "visibility") else None,
        }
    return kp


# --- Main: webcam + mediapipe pose ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

with mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
) as pose:
    frame_count = 0
    saved_keypoints = []  # accumulate some keypoints if you want to save later
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # draw skeleton
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # convert to pixel coords dict
            kp = keypoints_to_dict(results.pose_landmarks.landmark, frame.shape)

            # Example: compute left elbow angle (shoulder -> elbow -> wrist)
            try:
                left_shoulder = (kp["LEFT_SHOULDER"]["x"], kp["LEFT_SHOULDER"]["y"])
                left_elbow = (kp["LEFT_ELBOW"]["x"], kp["LEFT_ELBOW"]["y"])
                left_wrist = (kp["LEFT_WRIST"]["x"], kp["LEFT_WRIST"]["y"])
                left_elbow_angle = angle_between_points(
                    left_shoulder, left_elbow, left_wrist
                )
            except KeyError:
                left_elbow_angle = None

            # Example: compute right hip angle (shoulder -> hip -> knee)
            try:
                right_shoulder = (kp["RIGHT_SHOULDER"]["x"], kp["RIGHT_SHOULDER"]["y"])
                right_hip = (kp["RIGHT_HIP"]["x"], kp["RIGHT_HIP"]["y"])
                right_knee = (kp["RIGHT_KNEE"]["x"], kp["RIGHT_KNEE"]["y"])
                hip_angle = angle_between_points(right_shoulder, right_hip, right_knee)
            except KeyError:
                hip_angle = None

            # Simple feedback logic (example)
            feedback = []
            if left_elbow_angle is not None:
                feedback.append(f"Left elbow: {left_elbow_angle:.1f}°")
                if left_elbow_angle < 160:
                    feedback.append(" - Straighten left arm more")
            if hip_angle is not None:
                feedback.append(f"Hip angle: {hip_angle:.1f}°")
                if hip_angle < 40:
                    feedback.append(" - Raise your torso")

            # Overlay feedback text
            y0 = 30
            for i, text in enumerate(feedback):
                cv2.putText(
                    frame,
                    text,
                    (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Save sample keypoints every 30 frames
            if frame_count % 30 == 0:
                saved_keypoints.append({"frame": frame_count, "keypoints": kp})

        cv2.imshow("Pose Keypoints - Stretching Demo (press q to quit)", frame)
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# on exit save collected keypoints to a JSON file
with open("collected_keypoints.json", "w") as f:
    json.dump(saved_keypoints, f, indent=2)

cap.release()
cv2.destroyAllWindows()
print("Saved sample keypoints to collected_keypoints.json")
