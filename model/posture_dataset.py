"""
Posture Dataset for Training.

Provides a data generator class that loads JSON keypoints,
extracts features, and creates sliding window sequences
suitable for LSTM training.
"""

import csv
import glob
import os
import sys

import numpy as np

# Path setup for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow>=2.15.0")
    tf = None

try:
    from model.feature_extractor import NUM_FEATURES, extract_features_from_file
except ImportError:
    pass


class PostureDataset:
    """
    Dataset class for posture scoring model.

    Loads JSON keypoint files, matches them with labels from CSV,
    and generates sliding window sequences for LSTM training.

    If TensorFlow is available, can be used as tf.keras.utils.Sequence.
    Otherwise, provides a standard iterator interface.
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 30,
        batch_size: int = 32,
        shuffle: bool = True,
        overlap: float = 0.5,
    ):
        """
        Initialize the PostureDataset.

        Args:
            data_dir: Path to data directory (e.g., 'model/data/neck_stretch')
                      Expected structure:
                        data_dir/keypoint/*.json
                        data_dir/labels.csv
            sequence_length: Number of frames per sequence (sliding window size)
            batch_size: Batch size for training
            shuffle: Whether to shuffle data after each epoch
            overlap: Overlap ratio between consecutive windows (0.0 to 0.9)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.overlap = min(max(overlap, 0.0), 0.9)  # Clamp to valid range

        # Calculate stride from overlap
        self.stride = max(1, int(sequence_length * (1 - self.overlap)))

        # Load labels
        self.labels = self._load_labels()

        # Load and process all keypoint files
        self.samples = self._prepare_samples()

        # Create indices for batching
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_labels(self) -> dict:
        """
        Load labels from CSV file.

        Returns:
            Dict mapping filename (without extension) to score
        """
        labels = {}
        csv_path = os.path.join(self.data_dir, "labels.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: Labels file not found: {csv_path}")
            return labels

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                score = float(row.get("score", 0))
                if filename:
                    labels[filename] = score

        print(f"Loaded {len(labels)} labels from {csv_path}")
        return labels

    def _prepare_samples(self) -> list:
        """
        Prepare all samples by loading keypoints and creating windows.

        Returns:
            List of (features_window, score) tuples
        """
        samples = []
        keypoint_dir = os.path.join(self.data_dir, "keypoint")

        json_files = glob.glob(os.path.join(keypoint_dir, "*.json"))

        if not json_files:
            print(f"Warning: No JSON files found in {keypoint_dir}")
            return samples

        for json_path in json_files:
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(json_path))[0]

            # Get label for this file
            if filename not in self.labels:
                print(f"Warning: No label found for {filename}, skipping")
                continue

            score = self.labels[filename]

            # Extract features from file
            features = extract_features_from_file(json_path)
            num_frames = features.shape[0]

            if num_frames < self.sequence_length:
                print(
                    f"Warning: {filename} has only {num_frames} frames, "
                    f"need at least {self.sequence_length}, skipping"
                )
                continue

            # Create sliding windows
            for start in range(0, num_frames - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length
                window = features[start:end]
                samples.append((window, score / 100.0))  # Normalize score to 0-1

        print(f"Created {len(samples)} samples from {len(json_files)} files")
        return samples

    def __len__(self) -> int:
        """Return number of batches."""
        return max(1, len(self.samples) // self.batch_size)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a batch of data.

        Args:
            idx: Batch index

        Returns:
            (X, y) tuple where X is (batch_size, sequence_length, num_features)
            and y is (batch_size,)
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.samples))

        batch_indices = self.indices[start_idx:end_idx]

        X = np.array([self.samples[i][0] for i in batch_indices], dtype=np.float32)
        y = np.array([self.samples[i][1] for i in batch_indices], dtype=np.float32)

        return X, y

    def on_epoch_end(self):
        """Called at the end of each epoch. Shuffles indices if enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_all_data(self) -> tuple:
        """
        Get all data as numpy arrays.

        Useful for simple train/test splitting.

        Returns:
            (X, y) where X is (num_samples, sequence_length, num_features)
            and y is (num_samples,)
        """
        if not self.samples:
            return np.zeros((0, self.sequence_length, NUM_FEATURES)), np.zeros((0,))

        X = np.array([s[0] for s in self.samples], dtype=np.float32)
        y = np.array([s[1] for s in self.samples], dtype=np.float32)

        return X, y


# TensorFlow Sequence wrapper (if TensorFlow is available)
if tf is not None:

    class PostureSequence(tf.keras.utils.Sequence):
        """
        TensorFlow Keras Sequence wrapper for PostureDataset.

        Provides efficient data loading for model.fit().
        """

        def __init__(
            self,
            data_dir: str,
            sequence_length: int = 30,
            batch_size: int = 32,
            shuffle: bool = True,
            overlap: float = 0.5,
        ):
            """Initialize with same parameters as PostureDataset."""
            self.dataset = PostureDataset(
                data_dir=data_dir,
                sequence_length=sequence_length,
                batch_size=batch_size,
                shuffle=shuffle,
                overlap=overlap,
            )

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx: int) -> tuple:
            return self.dataset[idx]

        def on_epoch_end(self):
            self.dataset.on_epoch_end()


if __name__ == "__main__":
    # Quick test
    test_dir = "model/data/neck_stretch"

    print("Testing PostureDataset...")
    dataset = PostureDataset(test_dir, sequence_length=30, batch_size=8)
    print(f"Dataset length (batches): {len(dataset)}")

    if len(dataset) > 0:
        X, y = dataset[0]
        print(f"Batch X shape: {X.shape}")
        print(f"Batch y shape: {y.shape}")
        print(f"Sample y values: {y}")

        # Get all data
        X_all, y_all = dataset.get_all_data()
        print(f"\nAll data X shape: {X_all.shape}")
        print(f"All data y shape: {y_all.shape}")
