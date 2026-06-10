"""
calibration_tracker.py — exponential moving average trackers for drive calibration.

Each tracker holds one ratio and updates it with new measurements using EMA.
State machine owns the tracker instances and passes positions to calibration.py
to get new measurements, then hands them here.

Alpha = 0.15  (low weight on new samples — smooth but still adapts)
"""

ALPHA = 0.15
OUTLIER_THRESHOLD = 0.5   # reject if measurement differs by more than 50% from current ratio


class CalibrationTracker:
    def __init__(self, initial: float):
        self.ratio = initial

    def update(self, new_measurement: float) -> float:
        if abs(new_measurement - self.ratio) > self.ratio * OUTLIER_THRESHOLD:
            return self.ratio   # reject bad sample, keep current estimate
        self.ratio = ALPHA * new_measurement + (1 - ALPHA) * self.ratio
        return self.ratio


# One instance per motion type — import these into state_machine.py
from config import PIXELS_PER_ROTATION, DEGREES_PER_ROTATION

calibration_pixels = CalibrationTracker(PIXELS_PER_ROTATION)
calibration_angle  = CalibrationTracker(DEGREES_PER_ROTATION)
