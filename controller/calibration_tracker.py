"""
Trackers for drive and turn calibration.

Alpha = 0.15
"""

import os
import re

ALPHA = 0.15


class CalibrationTracker:
    def __init__(self, initial: float):
        self.ratio = initial

    def update(self, new_measurement: float) -> float:
        self.ratio = ALPHA * new_measurement + (1 - ALPHA) * self.ratio
        return self.ratio


# One instance per motion type, shared across the state machine and calibration manager.
from config import (
    BASE_DIR,
    PIXELS_PER_ROTATION,
    DEGREES_PER_ROTATION_LEFT,
    DEGREES_PER_ROTATION_RIGHT,
)

calibration_pixels      = CalibrationTracker(PIXELS_PER_ROTATION)
calibration_angle_left  = CalibrationTracker(DEGREES_PER_ROTATION_LEFT)
calibration_angle_right = CalibrationTracker(DEGREES_PER_ROTATION_RIGHT)


def save_calibration_to_config():
    """
    Called on ESC from main.py.
    Current calibration written into config.py as the new starting values on new runs.
    
    Returns (pixels_per_rotation, degrees_left, degrees_right).
    """
    new_values = {
        "PIXELS_PER_ROTATION":        calibration_pixels.ratio,
        "DEGREES_PER_ROTATION_LEFT":  calibration_angle_left.ratio,
        "DEGREES_PER_ROTATION_RIGHT": calibration_angle_right.ratio,
    }

    config_path = os.path.join(BASE_DIR, "config.py")
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()

    for name, value in new_values.items():
        # Match "NAME ... = <number>" at the start of a line, keep the original
        # "NAME ... = " prefix (and its alignment spacing), replace only the number.
        pattern = rf"^({re.escape(name)}\s*=\s*)[-+0-9.eE]+"
        text, count = re.subn(pattern, rf"\g<1>{value:.2f}", text, count=1, flags=re.M)
        if count == 0:
            raise ValueError(f"{name} not found in {config_path} - calibration not saved")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)

    return (new_values["PIXELS_PER_ROTATION"],
            new_values["DEGREES_PER_ROTATION_LEFT"],
            new_values["DEGREES_PER_ROTATION_RIGHT"])
