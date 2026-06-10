"""
calibration_tracker.py — EMA trackers for drive and turn calibration.

Each tracker holds one ratio and updates it with new measurements using EMA.
State machine owns the tracker instances; it passes measurements here after
each blocking move.

Alpha = 0.15  (low weight on new samples — smooth but still adapts)
"""

ALPHA = 0.15
OUTLIER_THRESHOLD = 2   # reject if measurement differs by >50% from current ratio


class CalibrationTracker:
    def __init__(self, initial: float):
        self.ratio = initial

    def update(self, new_measurement: float) -> float:
        if abs(new_measurement - self.ratio) > self.ratio * OUTLIER_THRESHOLD:
            print(f"[CAL] Outlier rejected: {new_measurement:.2f} (current {self.ratio:.2f})")
            return self.ratio
        self.ratio = ALPHA * new_measurement + (1 - ALPHA) * self.ratio
        return self.ratio


# One instance per motion type — import these into state_machine.py
from config import PIXELS_PER_ROTATION, DEGREES_PER_ROTATION

calibration_pixels = CalibrationTracker(PIXELS_PER_ROTATION)
calibration_angle  = CalibrationTracker(DEGREES_PER_ROTATION)
