"""Headless smoke test of the MoveUp-AI processing pipeline.

Runs: video frame -> MediaPipe Holistic -> extract_all_keypoints
      -> angle_between_points (right elbow) -> draw_pose_landmarks
      -> save annotated PNG so the user can eyeball it.

Does NOT touch Tkinter, the webcam, or the recording loop.
"""
import os
import sys
import json

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    angle_between_points,
    draw_pose_landmarks,
    extract_all_keypoints,
)

VIDEO = "model/data/neck_stretch/overlay/R.mp4"
OUT_IMG = "_pipeline_smoketest_out.png"

print(f"[1/5] Reading first frame from {VIDEO}")
cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"FAIL: could not read frame from {VIDEO}")
h, w = frame.shape[:2]
print(f"      frame shape = {w}x{h}")

print("[2/5] Running MediaPipe Holistic")
mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
) as holistic:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

detected = {
    "pose": results.pose_landmarks is not None,
    "face": results.face_landmarks is not None,
    "left_hand": results.left_hand_landmarks is not None,
    "right_hand": results.right_hand_landmarks is not None,
}
print(f"      detected: {detected}")
if not detected["pose"]:
    sys.exit("FAIL: no pose detected in frame — pipeline can't be verified end-to-end")

print("[3/5] Extracting keypoints via utils.extract_all_keypoints")
kp = extract_all_keypoints(results, frame.shape)
counts = {k: (len(v) if v else 0) for k, v in kp.items()}
print(f"      counts: {counts}")
assert counts["pose"] == 33, f"expected 33 pose landmarks, got {counts['pose']}"

print("[4/5] Computing right-elbow angle (shoulder-elbow-wrist)")
# MediaPipe Pose indices: 12=R_shoulder, 14=R_elbow, 16=R_wrist
p = kp["pose"]
shoulder = (p["12"]["x"], p["12"]["y"])
elbow    = (p["14"]["x"], p["14"]["y"])
wrist    = (p["16"]["x"], p["16"]["y"])
angle = angle_between_points(shoulder, elbow, wrist)
print(f"      right-elbow angle = {angle:.1f} deg")
assert 0 < angle < 180, f"angle out of range: {angle}"

print("[5/5] Drawing pose landmarks and saving annotated frame")
draw_pose_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
cv2.putText(
    frame,
    f"R-elbow {angle:.1f} deg",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 0),
    2,
)
cv2.imwrite(OUT_IMG, frame)
print(f"      wrote {OUT_IMG}")

print("\nOK — pipeline runs end-to-end on a real frame.")
print(f"Open {OUT_IMG} to eyeball the skeleton overlay.")
