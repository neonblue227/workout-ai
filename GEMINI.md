# MoveUp-AI

**AI-Powered Posture Detection for Healthy Stretching**

## Project Overview

MoveUp-AI is a computer vision application designed to help users perform upper body stretches correctly. It uses real-time webcam input to detect body pose landmarks (MediaPipe Holistic), calculate joint angles, and provide instant feedback on posture.

The core technology relies on **MediaPipe** for pose estimation and **OpenCV** for video processing and visualization.

## Architecture & Tech Stack

- **Language:** Python 3.8+
- **Computer Vision:** OpenCV (`cv2`), MediaPipe (`mp.solutions.holistic`)
- **Data Processing:** NumPy, Pandas, JSON
- **Analysis:** Jupyter Notebooks

## Project Structure

- `src/`: Contains the source code.
  - `record_keypoints.py`: Main application for high-precision video recording with real-time holistic keypoint overlay (Pose, Face, Hands). Saves video and JSON keypoints.
  - `generate_gif.py`: Generates animated GIFs from collected JSON keypoint data, visualizing body, hands, and face in a grid layout.
- `model/`: Machine learning models and analysis.
  - `classify.ipynb`: Jupyter notebook for pose classification experiments.
- `data/`: Directory for storing project data.
  - `gif/`: Generated demonstration GIFs.
  - `keypoint/`: JSON files containing extracted landmarks (Pose, Face, Hands).
  - `video/`: Raw recorded videos.
- `utils/`: Shared utility modules.
  - `angle.py`: Geometric calculations (e.g., angle between points).
  - `draw_*.py`: Visualization modules for angles, face mesh, hands, pose, and info overlays.
  - `keypoint_extractor.py`, `keypoint_recorder.py`: Logic for extracting and saving MediaPipe landmarks.
  - `gif_generator.py`: Core logic for creating GIFs.

## Setup & Installation

1.  **Environment Setup:**
    Ensure you have Python installed. It is recommended to use a virtual environment.

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Recording Videos & Keypoints

To start the high-precision recorder:

```bash
python src/record_keypoints.py
```

- **Functionality:** Opens the webcam, detects 553 landmarks (Pose, Face, Hands), and records both video (`.mp4`) and keypoint data (`.json`).
- **Features:** Real-time angle calculation, visibility-based coloring, and FPS calibration.
- **Controls:** Press `q` to stop recording early.
- **Output:** Files are saved to `data/video/` and `data/keypoint/`.

### Generating GIFs

To create visualizations from recorded data:

```bash
python src/generate_gif.py
```

- **Functionality:** Converts a JSON keypoint file into an animated GIF.
- **Input:** Prompts to select a JSON file from `data/keypoint/`.
- **Output:** Saves the GIF to `data/gif/` (or specified path).

## Development Notes

- **Holistic Tracking:** The project now uses MediaPipe Holistic to track Pose (33), Face (478), and Hands (21 each) simultaneously.
- **Modular Design:** Core logic is split into specific modules within `utils/` to separate visualization, calculation, and data management.
- **Data Format:** Keypoints are stored in JSON format, preserving the structure of MediaPipe landmarks for downstream analysis or playback.