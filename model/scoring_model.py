"""
Posture Scoring Model Architecture.

LSTM/GRU-based regression model for predicting posture quality scores
from sequences of skeletal keypoint features.
"""

import os
import sys

# Path setup for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow>=2.15.0")
    sys.exit(1)

try:
    from model.feature_extractor import NUM_FEATURES
except ImportError:
    pass


def create_scoring_model(
    sequence_length: int = 30,
    num_features: int = NUM_FEATURES,
    lstm_units: tuple = (64, 32),
    dropout_rate: float = 0.3,
    dense_units: int = 16,
    use_gru: bool = False,
) -> keras.Model:
    """
    Create LSTM/GRU regression model for posture scoring.

    Architecture:
        Input → LSTM(64) → Dropout → LSTM(32) → Dense(16) → Output(1)

    Args:
        sequence_length: Number of frames per sequence
        num_features: Number of features per frame (default 28)
        lstm_units: Tuple of units for each RNN layer
        dropout_rate: Dropout rate between layers (0.0-0.5)
        dense_units: Units in the dense layer before output
        use_gru: If True, use GRU instead of LSTM

    Returns:
        Compiled Keras model
    """
    RNNLayer = layers.GRU if use_gru else layers.LSTM

    # Input layer
    inputs = keras.Input(
        shape=(sequence_length, num_features), name="keypoint_sequence"
    )

    # First RNN layer (returns sequences for stacking)
    x = RNNLayer(
        units=lstm_units[0],
        return_sequences=True,
        name=f"{'gru' if use_gru else 'lstm'}_1",
    )(inputs)

    # Dropout for regularization
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    # Second RNN layer
    x = RNNLayer(
        units=lstm_units[1],
        return_sequences=False,
        name=f"{'gru' if use_gru else 'lstm'}_2",
    )(x)

    # Dense layer with ReLU
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)

    # Output layer - sigmoid activation scaled to 0-1 (multiply by 100 post-inference if needed)
    outputs = layers.Dense(1, activation="sigmoid", name="score_output")(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name="posture_scorer")

    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    loss: str = "mse",
) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        loss: Loss function ('mse' or 'mae')

    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["mae"],  # Mean Absolute Error as additional metric
    )

    return model


def create_callbacks(
    save_dir: str = "model/save",
    patience: int = 10,
    min_delta: float = 0.001,
) -> list:
    """
    Create training callbacks.

    Args:
        save_dir: Directory to save model checkpoints
        patience: Epochs to wait before early stopping
        min_delta: Minimum change to qualify as improvement

    Returns:
        List of Keras callbacks
    """
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        # Model checkpoint to save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "posture_scorer_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    return callbacks


def get_model_summary(model: keras.Model) -> str:
    """
    Get model summary as string.

    Args:
        model: Keras model

    Returns:
        Model summary string
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)


if __name__ == "__main__":
    import numpy as np

    print("Testing Posture Scoring Model...")

    # Create model
    model = create_scoring_model(sequence_length=30, num_features=28)
    model = compile_model(model)

    # Print summary
    print("\nModel Summary:")
    print(get_model_summary(model))

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = np.random.randn(4, 30, 28).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Output values: {predictions.flatten()}")

    print("\nModel created successfully!")
