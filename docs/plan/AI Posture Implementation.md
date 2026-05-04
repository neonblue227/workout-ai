# Implementation Plan: AI Posture Scoring Model (Regression)

## Context

**Project:** MoveUp-AI
**Goal:** Transition from basic pose detection to **Action Quality Assessment (AQA)**.
**Current Stack:** Python, MediaPipe Holistic, OpenCV, TensorFlow/Keras.
**Existing Assets:**

- `src/record_keypoints.py`: Records video and extracts JSON keypoints.
- `utils/angle.py`: Calculates geometric angles between landmarks.
- `model/data/neck_stretch/keypoint`: Stores the raw JSON landmark data for the neck_stretch posture.

## Objective

Develop a Deep Learning model (Regression) that accepts a sequence of skeletal keypoints (from MediaPipe) and outputs a **proficiency score (0-100)** indicating how correctly the user is performing the stretch.

---

## 🛠️ Step-by-Step Implementation Plan

### Phase 1: Data Preprocessing & Feature Engineering

**Goal:** Convert raw MediaPipe landmarks into invariant features suitable for temporal learning.

1.  **Data Loading:**

    - Load JSON files from `model/data/neck_stretch/keypoint`.
    - Load corresponding "Ground Truth" labels (a CSV file mapping `filename` to `score` [0-100]).

2.  **Feature Engineering (Crucial):**
    - Raw coordinates ($x, y, z$) are sensitive to camera distance. We must extract **invariant features**:
      - **Joint Angles:** Utilize `utils/angle.py` to calculate critical angles (e.g., Elbow, Shoulder, Hip, Knee).
      - **Relative Coordinates:** Normalize keypoints relative to a center point (e.g., `mid_hip`).
    - **Sequence Creation:** Apply a sliding window (e.g., 30 frames) to capture the _motion_ context, not just static poses.

### Phase 2: Model Architecture

**Goal:** Build a Time-Series Regression Model using TensorFlow/Keras.

- **Type:** RNN (LSTM or GRU).
- **Input Shape:** `(Sequence_Length, Num_Features)`
- **Architecture Draft:**
  - `Input Layer`
  - `LSTM/GRU` (e.g., 64 units, return_sequences=True) - capture temporal dynamics.
  - `Dropout` (0.2-0.5) - prevent overfitting.
  - `LSTM/GRU` (e.g., 32 units).
  - `Dense` (16 units, ReLU).
  - `Output Layer` (1 unit, Linear or Sigmoid scaled to 0-100) - predicts the score.

### Phase 3: Training Pipeline

**Goal:** Create a training script in `model/train_scoring_model.py` (or Jupyter Notebook).

- **Loss Function:** Mean Squared Error (MSE) or Mean Absolute Error (MAE).
- **Optimizer:** Adam.
- **Validation:** 80/20 Train-Test split.

### Phase 4: Inference & Integration

**Goal:** Integrate the trained model into the real-time application.

- Load the saved model (`.h5` or `.tflite`).
- Update `src/main.py` to:
  1. Maintain a buffer of the last $N$ frames of features.
  2. Pass the buffer to the model every $X$ frames.
  3. Display the predicted "Posture Score" on the UI overlay.

---

## 🤖 Prompt for Claude (Task Instruction)

**Role:** Senior AI Engineer / Computer Vision Specialist.

**Task:**
Please implement **Phase 1 and Phase 2** of the MoveUp-AI scoring system. I need you to write the Python code for a Data Generator and the Model Architecture.

**Requirements:**

1.  **Feature Extractor:** Create a function that takes raw MediaPipe landmarks and returns a vector of Features (Angles + Normalized Coordinates). Use the logic from `utils/angle.py` if applicable.
2.  **Dataset Class:** Create a `PostureDataset` class (inheriting from `tf.keras.utils.Sequence` or similar) that loads JSON keypoints and matches them with a dummy CSV label file (`filename`, `score`).
3.  **Model Definition:** Write the Keras code for the LSTM Regression model described in Phase 2.
4.  **Directory:** Assume the code will run in `model/` or `src/`.

**Note:** The model must output a continuous value (Regression), NOT a classification class. The score represents the quality of the stretch (0 to 100).
