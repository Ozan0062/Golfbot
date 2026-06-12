"""
calibration_manager.py — Deferred calibration after blocking moves.

After a drive or turn the EV3 brick blocks until the move finishes.  The
ArUco marker can't be read mid-move, so calibration is deferred: the state
machine records what it intended to do, then on the next frame (with fresh
pose data) CalibrationManager computes the actual result and updates the EMA
trackers in calibration_tracker.py.

Usage:
    cal = CalibrationManager()

    # before a drive:
    cal.record_drive(robot_px, rotations)

    # before a turn:
    cal.record_turn(robot_angle, rotations)

    # top of every frame (after pose is known):
    cal.consume(pose.px, pose.angle)
"""

from controller.calibration import measure_cm_per_rotation, measure_degrees_per_rotation
from controller.calibration_tracker import calibration_pixels, calibration_angle


class CalibrationManager:

    def __init__(self):
        self._pending_drive = None   # (start_px, rotations)
        self._pending_turn  = None   # (start_angle_deg, rotations)

    def record_drive(self, start_px: tuple, rotations: float):
        self._pending_drive = (start_px, rotations)

    def record_turn(self, start_angle: float, rotations: float):
        self._pending_turn = (start_angle, rotations)

    def consume(self, robot_px: tuple, robot_angle: float):
        """Apply any pending calibration using the current frame's pose."""
        if self._pending_drive is not None and robot_px is not None:
            start_px, rotations = self._pending_drive
            self._pending_drive = None
            if rotations > 0:
                measured = measure_cm_per_rotation(start_px, robot_px, rotations)
                calibration_pixels.update(measured)
                print(f"[CAL] px/rot → {calibration_pixels.ratio:.2f}")

        if self._pending_turn is not None and robot_angle is not None:
            start_angle, rotations = self._pending_turn
            self._pending_turn = None
            if rotations > 0:
                measured = measure_degrees_per_rotation(start_angle, robot_angle, rotations)
                calibration_angle.update(measured)
                print(f"[CAL] deg/rot → {calibration_angle.ratio:.2f}")
