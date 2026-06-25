"""
# before a drive:
cal.record_drive(robot_px, rotations)

# before a turn:
cal.record_turn(robot_angle, rotations)

# top of every frame (after pose is known):
cal.consume(pose.px, pose.angle)
"""

from controller.drive_calibration import measure_pixels_per_rotation, measure_degrees_per_rotation
from controller.calibration_tracker import calibration_pixels, calibration_angle_left, calibration_angle_right
from controller.zone_calibration_tracker import zone_tracker, get_zone
from golfbot_logger import get_logger

log = get_logger(__name__)


MIN_DRIVE_ROTATIONS = 0.5   # ignore drive measurements below this
MIN_TURN_ROTATIONS  = 0.3   # ignore turn measurements below this
MIN_DRIVE_FRACTION  = 0.5   # discard if robot moved less than this fraction of expected distance
                            # (indicates it hit a wall or obstacle and stalled)


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
                import math
                actual_px   = math.dist(start_px, robot_px)
                expected_px = rotations * calibration_pixels.ratio
                if actual_px < expected_px * MIN_DRIVE_FRACTION:
                    log.debug("Drive calibration skipped - moved %.1f px, expected %.1f px (likely stalled)",
                              actual_px, expected_px)
                else:
                    measured = measure_pixels_per_rotation(start_px, robot_px, rotations)
                    # Global tracker
                    calibration_pixels.update(measured)
                    log.debug("calibrated px/rot -> %.2f", calibration_pixels.ratio)
                    # Zone tracker - only if start and end are in the same zone
                    start_zone = get_zone(start_px, zone_tracker.center_px)
                    end_zone   = get_zone(robot_px, zone_tracker.center_px)
                    if start_zone is not None and start_zone == end_zone:
                        zone_tracker.update_drive(start_zone, measured)

        if self._pending_turn is not None and robot_angle is not None:
            start_angle, rotations, direction = self._pending_turn
            self._pending_turn = None
            if rotations >= MIN_TURN_ROTATIONS:
                measured = measure_degrees_per_rotation(start_angle, robot_angle, rotations)
                # Global tracker
                tracker = calibration_angle_left if direction == "LEFT" else calibration_angle_right
                tracker.update(measured)
                log.debug("calibrated deg/rot %s -> %.2f", direction, tracker.ratio)
                # Zone tracker - turns stay in-place, so use the current position
                if robot_px is not None:
                    zone = get_zone(robot_px, zone_tracker.center_px)
                    if zone is not None:
                        zone_tracker.update_turn(zone, measured, direction)
