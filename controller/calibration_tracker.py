"""
calibration_tracker.py — EMA trackers for drive and turn calibration.

Each tracker holds one ratio and updates it with new measurements using EMA.
State machine owns the tracker instances; it passes measurements here after
each blocking move.

Alpha = 0.15
"""

ALPHA = 0.15


class CalibrationTracker:
    def __init__(self, initial: float):
        self.ratio = initial

    def update(self, new_measurement: float) -> float:
        self.ratio = ALPHA * new_measurement + (1 - ALPHA) * self.ratio
        return self.ratio


# One instance per motion type, shared across the state machine and calibration manager.
from config import PIXELS_PER_ROTATION, DEGREES_PER_ROTATION

calibration_pixels = CalibrationTracker(PIXELS_PER_ROTATION)
calibration_angle  = CalibrationTracker(DEGREES_PER_ROTATION)
