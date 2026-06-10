"""
calibration.py — math helpers for drive/turn calibration.
Called after every movement to refine calibration_tracker.py estimates.
"""

import math


# --- Measurement: call after each move, pass result to calibration_tracker ---

def measure_pixels_per_rotation(start_px, end_px, rotations: float) -> float:
    """Pixels travelled per rotation from a forward/backward move."""
    return math.dist(start_px, end_px) / rotations


def measure_cm_per_rotation(start_cm, end_cm, rotations: float) -> float:
    """CM travelled per rotation from a forward/backward move (world coords in cm)."""
    return math.dist(start_cm, end_cm) / rotations


def measure_degrees_per_rotation(start_heading: float, end_heading: float, rotations: float) -> float:
    """Degrees turned per rotation from a left/right turn."""
    delta = (end_heading - start_heading + 180) % 360 - 180  # shortest path, handles wrap-around
    return abs(delta) / rotations


# --- Runtime: convert desired distance/angle into rotations to command -------

def rotations_for_distance(distance_px: float, pixels_per_rotation: float) -> float:
    return distance_px / pixels_per_rotation


def rotations_for_angle(angle_deg: float, degrees_per_rotation: float) -> float:
    return abs(angle_deg) / degrees_per_rotation
