"""
Translates pixels and angles into motor rotations.
Also contains the Driver class that sends commands to the robot.
"""

import math

import controller.ev3_controller as robot
from controller.commands import Command
from controller.navigation import (
    angle_to_target, angle_error, staging_point, px_to_cm,
    classify_zone, wall_approach_angle, cross_approach_angle,
    cross_trigger_radius, path_is_clear, obstacle_waypoint,
)
from controller.calibration_tracker import (
    calibration_pixels, calibration_angle_left, calibration_angle_right,
)
from controller.zone_calibration_tracker import zone_tracker
from config import (
    ALIGN_THRESHOLD_DEG, MIN_TURN_ROTATIONS, TURN_DAMPING, MAX_DRIVE_PX,
    WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT, CORNER_STAGE_DISTANCES_PX,
    FIELD_EDGE_MARGIN_PX, CROSS_CLEARANCE_PX, AVOID_WAYPOINT_DIST_PX,
    CROSS_RADIUS_PX,
)
from golfbot_logger import get_logger

log = get_logger(__name__)


# --- Unit conversions --------------------------------------------------------

def distance_px(a, b):
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def px_to_rotations(drive_px, pos_px=None):
    """Convert pixels to motor rotations using zone calibration if possible."""
    if pos_px is not None:
        ratio = zone_tracker.get_px_per_rotation(pos_px)
    else:
        ratio = calibration_pixels.ratio
    return drive_px / ratio


def angle_to_rotations(heading_error, pos_px=None):
    """Convert degrees to motor rotations using zone calibration if possible."""
    direction = "RIGHT" if heading_error > 0 else "LEFT"
    if pos_px is not None:
        ratio = zone_tracker.get_deg_per_rotation(pos_px, direction)
    else:
        tracker = calibration_angle_right if heading_error > 0 else calibration_angle_left
        ratio = tracker.ratio
    return abs(heading_error) / ratio * TURN_DAMPING


# --- Staging geometry --------------------------------------------------------

def corner_approach_waypoints(robot_px, ball_px, approach_angle_deg,
                              stage_distances, field_w, field_h, margin):
    """
    Ordered waypoints along the approach axis for a wall/corner (or goal) ball.

    All stages are always returned (far->close), regardless of the robot's
    current position. Each waypoint is clamped to stay inside the field boundary.
    """
    waypoints = []
    for dist in stage_distances:                # already ordered far->close
        sp = staging_point(ball_px, approach_angle_deg, dist)
        sp = (
            max(margin, min(sp[0], field_w - margin)),
            max(margin, min(sp[1], field_h - margin)),
        )
        waypoints.append(sp)
    return waypoints


# --- Route-cost simulation (TSP edge weight) ---------------------------------

def _approach_waypoints(start_px, target_px):
    """Get the final approach waypoints based on where the ball is."""
    zone, walls = classify_zone(target_px, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
    if zone in ("wall", "corner"):
        angle = wall_approach_angle(walls)
        if angle is not None:
            stages = corner_approach_waypoints(
                start_px, target_px, angle, CORNER_STAGE_DISTANCES_PX,
                WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
            )
            return stages + [target_px]
    return [target_px]


def plan_route_waypoints(start_px, target_px, cross_px=None, cross_size_px=None):
    if cross_px is not None:
        # 1. Ball at the cross: approach like a corner, in along a fixed diagonal.
        radius = cross_trigger_radius(CROSS_RADIUS_PX)
        if distance_px(target_px, cross_px) <= radius:
            angle = cross_approach_angle(target_px, cross_px)
            stages = corner_approach_waypoints(
                start_px, target_px, angle, CORNER_STAGE_DISTANCES_PX,
                WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
            )
            return stages + [target_px]

        # 2. Cross blocks the straight path: steer around it, then approach.
        clear, _ = path_is_clear(start_px, target_px, [cross_px], CROSS_CLEARANCE_PX)
        if not clear:
            wp = obstacle_waypoint(start_px, target_px, cross_px,
                                   AVOID_WAYPOINT_DIST_PX, WARPED_WIDTH, WARPED_HEIGHT)
            if wp is not None:
                # FSM re-plans from the dodge waypoint; the cross is now out of the
                # way, so finish with the normal wall/open approach from there.
                return [wp] + _approach_waypoints(wp, target_px)

    # 3 & 4. Wall/corner staging or a straight open-field drive.
    return _approach_waypoints(start_px, target_px)


def get_price(start_px, target_px, *, cross_px=None, cross_size_px=None, start_angle_deg=None):
    if start_px is None or target_px is None:
        raise ValueError("get_price needs concrete (x, y) pixel coordinates")

    waypoints = plan_route_waypoints(start_px, target_px, cross_px, cross_size_px)

    total_rotations = 0.0
    prev_px = start_px
    heading = start_angle_deg

    for wp in waypoints:
        # Drive this leg.
        total_rotations += px_to_rotations(distance_px(prev_px, wp), pos_px=prev_px)

        # Turn onto this leg. Bearing in cm so the warp doesn't distort it.
        desired = angle_to_target(px_to_cm(prev_px), px_to_cm(wp))
        if heading is not None and abs(angle_error(heading, desired)) > ALIGN_THRESHOLD_DEG:
            total_rotations += angle_to_rotations(angle_error(heading, desired), pos_px=prev_px)
        heading = desired
        prev_px = wp

    return total_rotations


# --- Driver ------------------------------------------------------------------

class Driver:
    """Helper class to send motor commands and update calibration."""

    def __init__(self, cal, pose_cache):
        self._cal  = cal
        self._pose = pose_cache

    def drive(self, pose, rotations):
        rotations = max(rotations, 0.1)
        self._cal.record_drive(pose.px, rotations)
        robot.drive(rotations)
        self._pose.invalidate()

    def turn(self, pose, rotations, direction):
        rotations = max(rotations, 0.1)
        self._cal.record_turn(pose.angle, rotations, direction.name)
        robot.turn(rotations, direction.name)
        self._pose.invalidate()

    def reverse(self, rotations):
        robot.reverse(rotations)
        self._pose.invalidate()

    def drive_toward(self, pose, target_px, arrive_radius):
        """
        One movement step toward target_px: turn to face it, or drive toward it
        if already aligned. Returns (command, arrived) where arrived is True
        once the robot is within arrive_radius (no command issued in that case).
        """
        dist = distance_px(pose.px, target_px)
        if dist <= arrive_radius:
            return Command.STOP, True

        # Bearing in cm (not raw pixels) so it matches the cm-frame heading.
        heading_error = angle_error(
            pose.angle, angle_to_target(px_to_cm(pose.px), px_to_cm(target_px))
        )
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            rotations = angle_to_rotations(heading_error, pos_px=pose.px)
            if rotations >= MIN_TURN_ROTATIONS:
                direction = Command.RIGHT if heading_error > 0 else Command.LEFT
                log.debug("turn %s %.1f deg -> %.2f rot", direction.name, abs(heading_error), rotations)
                self.turn(pose, rotations, direction)
                return direction, False

        drive_px = min(dist - arrive_radius, MAX_DRIVE_PX)
        log.debug("drive %.0f px", drive_px)
        self.drive(pose, px_to_rotations(drive_px, pos_px=pose.px))
        return Command.FORWARD, False
