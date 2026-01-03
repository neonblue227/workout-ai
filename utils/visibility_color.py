"""
Visibility-based color utility for landmark visualization.

Provides color coding based on landmark detection confidence levels.
"""


def get_visibility_color(visibility):
    """
    Return BGR color based on landmark visibility.

    Color coding:
        - Green (0, 255, 0): High confidence (visibility > 0.8)
        - Yellow (0, 255, 255): Medium confidence (visibility > 0.5)
        - Red (0, 0, 255): Low confidence (visibility <= 0.5)

    Args:
        visibility: Float between 0.0 and 1.0 indicating detection confidence

    Returns:
        tuple: BGR color tuple (B, G, R)
    """
    if visibility > 0.8:
        return (0, 255, 0)  # Green - high confidence
    elif visibility > 0.5:
        return (0, 255, 255)  # Yellow - medium confidence
    else:
        return (0, 0, 255)  # Red - low confidence
