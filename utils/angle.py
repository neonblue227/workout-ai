"""
Angle calculation utility for pose analysis.

Provides a function to calculate the angle between three points,
commonly used for joint angle measurements in pose estimation.
"""

from math import acos, degrees


def angle_between_points(a, b, c):
    """
    Calculate angle (degrees) at point b formed by a-b-c.

    Args:
        a: First point as (x, y) tuple
        b: Vertex point as (x, y) tuple (angle is measured here)
        c: Third point as (x, y) tuple

    Returns:
        float: Angle in degrees, or None if calculation is not possible
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    norm2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return None
    cosang = max(-1.0, min(1.0, dot / (norm1 * norm2)))

    return degrees(acos(cosang))
