# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoveUp-AI is a computer vision application for real-time posture detection during stretching exercises. It uses MediaPipe for pose estimation and OpenCV for video processing, providing instant feedback on body posture via joint angle calculations.

## Commands

### Run Main Application
```bash
python src/main.py
```
Opens webcam, performs pose detection, displays skeleton overlay with angle feedback. Press `q` to quit. Saves keypoints to `collected_keypoints.json` on exit.

### Record Training Videos
```bash
python src/video_record_pg.py
```
Interactive CLI prompts for filename, resolution, duration, and format. Videos saved to `data/video/`.

### Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Process Keypoint Data
Run `src/data_tranfrom.ipynb` in Jupyter to batch process videos and extract keypoints to JSON.

## Architecture

**Processing Pipeline:**
1. **Input:** Webcam frame capture via OpenCV
2. **Detection:** MediaPipe pose estimation extracts 33 body landmarks
3. **Normalization:** Landmarks converted from normalized (0-1) to pixel coordinates
4. **Analysis:** Joint angles calculated using 3-point geometry (e.g., shoulder-elbow-wrist)
5. **Output:** Real-time skeleton visualization with angle-based feedback overlay

**Key Functions in `src/main.py`:**
- `normalized_to_pixel()` - Converts MediaPipe normalized coords to pixel space
- `angle_between_points()` - Calculates angle between 3 landmarks
- `keypoints_to_dict()` - Structures landmarks into named dictionary

**Data Flow:**
- Videos stored in `data/video/`
- Extracted keypoints saved as JSON in `data/keypoint/`
- Visualization GIFs generated in `data/gif/`

## Tech Stack

- **Core:** Python 3.8+, MediaPipe 0.10.x, OpenCV
- **Analysis:** NumPy, Matplotlib, Jupyter
- **Data Format:** JSON for keypoint storage

## Project Status

Currently in Phase 1 (Foundation). Core pose detection works. Pending: stretch detection algorithms, posture classification model, web interface (Flask/FastAPI planned).