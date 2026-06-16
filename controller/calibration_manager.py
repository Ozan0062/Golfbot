"""
# before a drive:
cal.record_drive(robot_px, rotations)

# before a turn:
cal.record_turn(robot_angle, rotations)

# top of every frame (after pose is known):
cal.consume(pose.px, pose.angle)
"""

from controller.calibration import measure_pixels_per_rotation, measure_degrees_per_rotation
from controller.calibration_tracker import calibration_pixels, calibration_angle_left, calibration_angle_right


MIN_DRIVE_ROTATIONS = 0.5   # ignore drive measurements below this
MIN_TURN_ROTATIONS  = 0.3   # ignore turn measurements below this


class CalibrationManager:

    def __init__(self):
        self._pending_drive = None   # (start_px, rotations)
        self._pending_turn  = None   # (start_angle_deg, rotations, direction)

    def record_drive(self, start_px: tuple, rotations: float):
        self._pending_drive = (start_px, rotations)

    def record_turn(self, start_angle: float, rotations: float, direction: str):
        """direction: 'LEFT' or 'RIGHT'"""
        self._pending_turn = (start_angle, rotations, direction)

    def consume(self, robot_px: tuple, robot_angle: float):
        """Apply any pending calibration using the current frame's pose."""
        if self._pending_drive is not None and robot_px is not None:
            start_px, rotations = self._pending_drive
            self._pending_drive = None
            if rotations >= MIN_DRIVE_ROTATIONS:
                measured = measure_pixels_per_rotation(start_px, robot_px, rotations)
                calibration_pixels.update(measured)
                print(f"[CAL] px/rot → {calibration_pixels.ratio:.2f}")

        if self._pending_turn is not None and robot_angle is not None:
            start_angle, rotations, direction = self._pending_turn
            self._pending_turn = None
            if rotations >= MIN_TURN_ROTATIONS:
                measured = measure_degrees_per_rotation(start_angle, robot_angle, rotations)
                tracker = calibration_angle_left if direction == "LEFT" else calibration_angle_right
                tracker.update(measured)
                print(f"[CAL] deg/rot {direction} → {tracker.ratio:.2f}")
