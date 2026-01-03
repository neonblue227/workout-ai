# MoveUp-AI

**AI-Powered Posture Detection for Healthy Stretching**

## Project Overview

MoveUp-AI is a computer vision application designed to help users perform upper body stretches correctly. It uses real-time webcam input to detect body pose landmarks, calculate joint angles, and provide instant feedback on posture.

The core technology relies on **MediaPipe** for pose estimation and **OpenCV** for video processing and visualization.

## Architecture & Tech Stack

- **Language:** Python 3.8+
- **Computer Vision:** OpenCV (`cv2`), MediaPipe (`mp.solutions.pose`)
- **Data Processing:** NumPy, Pandas, JSON
- **Analysis:** Jupyter Notebooks

## Project Structure

- `src/`: Contains the source code.
  - `main.py`: The main entry point. Captures webcam feed, runs pose detection, calculates angles (e.g., elbow, hip), and displays feedback overlays.
  - `video_record_pg.py`: A utility script for recording training or testing videos with customizable resolution and duration.
  - `data_tranfrom.ipynb`: (Note typo in filename, referenced as `data_transform.ipynb` in README) Jupyter notebook for analyzing or transforming collected keypoint data.
- `data/`: Directory for storing project data.
  - `gif/`: Demonstration GIFs.
  - `keypoint/`: JSON files containing extracted pose landmarks.
  - `video/`: Recorded videos.
- `utils/`: Helper utilities (currently empty or used for internal modules).

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

### Running the Main Application

To start the real-time posture detection:

```bash
python src/main.py
```

- **Functionality:** Opens the webcam, draws the skeleton overlay, and displays angle calculations (e.g., Left Elbow, Hip).
- **Controls:** Press `q` to quit the application.
- **Output:** Upon exit, it saves collected keypoints to `collected_keypoints.json` in the root directory.

### Recording Videos

To record a new video sample (e.g., for training or testing):

```bash
python src/video_record_pg.py
```

- **Prompts:** You will be asked to specify:
  1.  Filename prefix (default: 'recording')
  2.  Resolution (e.g., 640x480, 1280x720)
  3.  Duration (seconds)
  4.  Format (.avi or .mp4)
- **Storage:** Videos are automatically saved to `data/video/` with an incrementing counter (e.g., `recording_1.mp4`, `recording_2.mp4`).

## Development Notes

- **Keypoint Extraction:** `main.py` converts normalized MediaPipe landmarks (0.0-1.0) to pixel coordinates based on the frame size.
- **Angle Calculation:** The `angle_between_points` function in `main.py` computes the angle between three landmarks (e.g., Shoulder-Elbow-Wrist).
- **Conventions:**
  - Python code follows standard PEP 8 style.
  - Data files are organized by type in the `data/` directory.
