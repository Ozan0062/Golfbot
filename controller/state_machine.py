"""
state_machine.py -- GolfBot decision logic.

STATE FLOW
----------
  SEEK  ->  lock onto next ball from TSP route
    | path obstructed by cross?
    | yes -> AVOID  ->  drive to waypoint (2x clearance from cross, perpendicular)
    |                   then back to SEEK (original target still locked)
    | ball near wall/corner?
    | yes -> AVOID  ->  drive to staging point (60px back, on approach line)
    |                   then back to SEEK -> ALIGN is now perpendicular to wall
    | no
  ALIGN  ->  turn to face the locked target
    | heading within 10 deg
  APPROACH  ->  drive toward target (capped per step)
    | after each drive step, go back to ALIGN
    | when within collect radius
  (collect ball, clear target, back to SEEK)

  WALL/CORNER APPROACH
  --------------------
  Wall ball (within 30px of one wall):
    staging point 60px from ball, perpendicular to wall
    robot aligns perpendicular, drives straight in, collects
  Corner ball (within 30px of two walls):
    staging point 60px from ball, at 45 deg diagonal
    robot aligns on diagonal, drives straight in, collects

  When SEEK finds no balls:
  REVERSE_WHITE  ->  back up, re-scan for white or orange
    | still nothing
  REVERSE_ORANGE  ->  back up, re-scan for orange only
    | still nothing (all collected)
  DRIVE_GOAL  ->  turn + drive to goal zone
    | arrived
  RELEASE  ->  open gate
    |
  DONE

All movement maths (angles, distances) live in navigation.py.
All hardware commands live in ev3_controller.py.
All calibration lives in calibration_manager.py + calibration_tracker.py.
"""

from enum import Enum, auto
import math

import controller.ev3_controller as robot
from controller.calibration_manager import CalibrationManager
from controller.calibration_tracker import calibration_angle, calibration_pixels
from controller.commands import Command
from controller.navigation import (
    angle_to_target, angle_error, path_is_clear, obstacle_waypoint,
    classify_zone, wall_approach_angle, staging_point,
)
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager, RouteTarget
from config import GOAL_POSITION_CM, GOAL_POSITION_PX, WARPED_WIDTH, WARPED_HEIGHT


# --- Thresholds --------------------------------------------------------------

ALIGN_THRESHOLD_DEG = 10    # Below this, we consider ourselves "aligned" and drive straight.
MIN_TURN_ROTATIONS = 0.25   # Ignore turns smaller than this.
TURN_DAMPING = 0.6          # Reduce turn commands to prevent oscillation when close.

COLLECT_RADIUS_PX = 15      # Accepted radius for collecting a ball.
GOAL_THRESHOLD_PX = 30      # Close enough to goal to stop driving and release balls.
REVERSE_ROTATIONS = 1.5     # How far to reverse when no balls are visible.
MAX_DRIVE_PX = 80           # Cap on drive distance per cycle to allow for course correction.

# DECISION: The cross minimum distance is the closest the robot's path may
# pass to the cross without triggering avoidance. The waypoint is placed at
# 2x this distance so the robot clears the cross with comfortable margin.
CROSS_CLEARANCE_PX = 30     # Min distance from cross before avoidance triggers.
AVOID_WAYPOINT_DIST_PX = CROSS_CLEARANCE_PX * 2   # Waypoint offset from cross.
AVOID_ARRIVE_PX = 20        # Close enough to waypoint to consider it reached.

# DECISION: Balls within WALL_MARGIN_PX of an edge are "wall balls" and need
# a perpendicular approach so the flat claw can reach without the robot body
# hitting the wall. Corner balls (near two walls) use a 45 deg diagonal.
# The robot first drives to a staging point STAGING_DISTANCE_PX behind the
# ball along the approach line, then the normal ALIGN->APPROACH loop drives
# straight in at the correct angle.
WALL_MARGIN_PX = 30         # Ball this close to a wall triggers wall approach.
STAGING_DISTANCE_PX = 60    # How far back from the ball the staging point is.


# --- States -------------------------------------------------------------------

class State(Enum):
    SEEK           = auto()   # acquire next target from TSP route
    AVOID          = auto()   # drive to waypoint to get around cross
    ALIGN          = auto()   # turn to face the locked target
    APPROACH       = auto()   # drive toward the locked target
    REVERSE_WHITE  = auto()   # back up, scan for white or orange balls
    REVERSE_ORANGE = auto()   # back up, scan for orange ball only
    DRIVE_GOAL     = auto()   # navigate to goal zone (turn + drive)
    RELEASE        = auto()   # dump balls at goal
    DONE           = auto()   # mission complete


# --- Controller ---------------------------------------------------------------

class GolfBotController:

    def __init__(self):
        self.state           = State.SEEK
        self._pose           = PoseCache()
        self._route          = RouteManager()
        self._cal            = CalibrationManager()
        self._has_reversed   = False
        self._locked_target  = None   # type: RouteTarget | None
        self._avoid_target   = None   # type: tuple
        print(f"[FSM] Ready.  Goal={GOAL_POSITION_CM}  "
              f"cal={calibration_angle.ratio:.1f}deg/rot  {calibration_pixels.ratio:.1f}px/rot")

    # -- Main entry point (called once per camera frame) -----------------------

    def update(self, world: dict) -> Command:
        """Run one tick of the state machine. Returns the command executed."""
        pose = self._pose.update(world)
        if pose is None:
            print("[FSM] ArUco not detected -- waiting.")
            return Command.STOP

        self._cal.consume(pose.px, pose.angle)

        # -- State dispatch ----------------------------------------------------
        if self.state == State.SEEK:
            return self._seek(pose, world)
        if self.state == State.AVOID:
            return self._avoid(pose)
        if self.state == State.ALIGN:
            return self._align(pose)
        if self.state == State.APPROACH:
            return self._approach(pose)
        if self.state == State.REVERSE_WHITE:
            return self._handle_reverse(world, scan_for=("white_balls", "ob"),
                                        next_if_found=State.SEEK,
                                        next_if_empty=State.REVERSE_ORANGE)
        if self.state == State.REVERSE_ORANGE:
            return self._handle_reverse(world, scan_for=("ob",),
                                        next_if_found=State.SEEK,
                                        next_if_empty=State.DRIVE_GOAL)
        if self.state == State.DRIVE_GOAL:
            return self._drive_to_goal(pose)
        if self.state == State.RELEASE:
            return self._release_balls()
        return Command.STOP   # DONE

    # --- State: SEEK ----------------------------------------------------------
    # Lock onto the next ball.
    # 1. Check if the path is obstructed by the cross -> AVOID waypoint
    # 2. Check if ball is near wall/corner -> AVOID to staging point
    # 3. Otherwise -> ALIGN directly

    def _seek(self, pose, world) -> Command:
        # Re-lock target only if we don't already have one (returning from AVOID)
        if self._locked_target is None:
            self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        if self._locked_target is None:
            print("[FSM] No balls visible -- reversing.")
            self._route.clear()
            self._has_reversed = False
            self._transition(State.REVERSE_WHITE)
            return Command.STOP

        target = self._locked_target
        print(f"[SEEK] Target at ({target.px[0]:.0f},{target.px[1]:.0f})px")

        # -- Check if cross blocks the path ------------------------------------
        cross_px = world.get("cross_px")
        if cross_px is not None:
            clear, _ = path_is_clear(pose.px, target.px, [cross_px],
                                     CROSS_CLEARANCE_PX)
            if not clear:
                wp = obstacle_waypoint(pose.px, target.px, cross_px,
                                       AVOID_WAYPOINT_DIST_PX,
                                       WARPED_WIDTH, WARPED_HEIGHT)
                if wp is not None:
                    self._avoid_target = wp
                    print(f"[SEEK] Cross blocking path -- avoid via "
                          f"({wp[0]:.0f},{wp[1]:.0f})px")
                    self._transition(State.AVOID)
                    return Command.STOP

        # -- Check if ball is near a wall or corner -----------------------------
        zone, walls = classify_zone(target.px, WALL_MARGIN_PX,
                                    WARPED_WIDTH, WARPED_HEIGHT)
        if zone in ("wall", "corner"):
            angle = wall_approach_angle(walls)
            if angle is not None:
                sp = staging_point(target.px, angle, STAGING_DISTANCE_PX)
                dist_to_staging = _distance_px(pose.px, sp)
                if dist_to_staging > AVOID_ARRIVE_PX:
                    self._avoid_target = sp
                    print(f"[SEEK] {zone} ball (walls={walls}) -- "
                          f"staging at ({sp[0]:.0f},{sp[1]:.0f})px")
                    self._transition(State.AVOID)
                    return Command.STOP
                # Already at staging point -- fall through to ALIGN
                print(f"[SEEK] Already at staging point for {zone} ball")

        # Path is clear -- proceed to alignment
        self._transition(State.ALIGN)
        return Command.STOP

    # --- State: AVOID ---------------------------------------------------------
    # Turn and drive to the avoidance waypoint (same pattern as DRIVE_GOAL).
    # When reached, go back to SEEK -- the original _locked_target is preserved,
    # and SEEK will re-check if the path is now clear.

    def _avoid(self, pose) -> Command:
        wp = self._avoid_target
        if wp is None:
            self._transition(State.SEEK)
            return Command.STOP

        # Turn to face waypoint
        heading_error = self._heading_error_to_px(pose, wp)
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            rotations = _angle_to_rotations(heading_error)
            if rotations >= MIN_TURN_ROTATIONS:
                direction = Command.RIGHT if heading_error > 0 else Command.LEFT
                print(f"[AVOID] Turn {direction.name}  "
                      f"{abs(heading_error):.1f}deg -> {rotations:.2f}rot")
                self._execute_turn(pose, rotations, direction)
                return direction

        # Drive toward waypoint
        dist_px = _distance_px(pose.px, wp)
        if dist_px > AVOID_ARRIVE_PX:
            drive_px  = min(dist_px - AVOID_ARRIVE_PX, MAX_DRIVE_PX)
            rotations = _px_to_rotations(drive_px)
            print(f"[AVOID] Drive {drive_px:.0f}px -> {rotations:.2f}rot")
            self._execute_drive(pose, rotations)
            return Command.FORWARD

        # Arrived at waypoint -- clear it and re-check path from new position
        print("[AVOID] Waypoint reached -- returning to SEEK")
        self._avoid_target = None
        self._transition(State.SEEK)
        return Command.STOP

    # --- State: ALIGN ---------------------------------------------------------
    # Turn to face the locked target. Once heading error is within threshold,
    # transition to APPROACH. Each frame either turns or passes through.

    def _align(self, pose) -> Command:
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        heading_error = self._heading_error_to(pose, target.cm)
        print(f"[ALIGN] heading={pose.angle:.1f}deg  error={heading_error:.1f}deg")

        if abs(heading_error) <= ALIGN_THRESHOLD_DEG:
            self._transition(State.APPROACH)
            return Command.STOP

        rotations = _angle_to_rotations(heading_error)
        if rotations < MIN_TURN_ROTATIONS:
            self._transition(State.APPROACH)
            return Command.STOP

        direction = Command.RIGHT if heading_error > 0 else Command.LEFT
        print(f"[ALIGN] Turn {direction.name}  {abs(heading_error):.1f}deg -> {rotations:.2f}rot")
        self._execute_turn(pose, rotations, direction)
        return direction
        # Stay in ALIGN -- next frame will re-check heading

    # --- State: APPROACH ------------------------------------------------------
    # Drive toward the locked target. After each drive step, go back to ALIGN
    # to re-check heading. When close enough, collect and return to SEEK.

    def _approach(self, pose) -> Command:
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        dist_px = _distance_px(pose.px, target.px)
        print(f"[APPROACH] dist={dist_px:.0f}px  target=({target.px[0]:.0f},{target.px[1]:.0f})px")

        # Close enough -> collect
        if dist_px <= COLLECT_RADIUS_PX:
            print(f"[APPROACH] Within collect radius -- grabbing ball")
            self._locked_target = None
            self._route.advance()
            robot.collect()
            self._pose.invalidate()
            self._transition(State.SEEK)
            return Command.COLLECT

        # Drive forward (capped), then re-align
        drive_px  = min(dist_px - COLLECT_RADIUS_PX, MAX_DRIVE_PX)
        rotations = _px_to_rotations(drive_px)
        print(f"[APPROACH] Drive {drive_px:.0f}px -> {rotations:.2f}rot")
        self._execute_drive(pose, rotations)
        self._transition(State.ALIGN)
        return Command.FORWARD
        # Next frame enters ALIGN to re-check heading before driving again

    # --- State: REVERSE (shared by REVERSE_WHITE and REVERSE_ORANGE) ----------

    def _handle_reverse(self, world, scan_for, next_if_found, next_if_empty):
        if not self._has_reversed:
            robot.reverse(REVERSE_ROTATIONS)
            self._pose.invalidate()
            self._has_reversed = True
            return Command.BACKWARD

        self._has_reversed = False
        if any(world.get(key) for key in scan_for):
            self._transition(next_if_found)
        else:
            self._transition(next_if_empty)
        return Command.STOP

    # --- State: DRIVE_GOAL ----------------------------------------------------
    # Same turn-then-drive pattern, but toward the fixed goal position.

    def _drive_to_goal(self, pose) -> Command:
        heading_error = self._heading_error_to(pose, GOAL_POSITION_CM)

        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            rotations = _angle_to_rotations(heading_error)
            if rotations >= MIN_TURN_ROTATIONS:
                direction = Command.RIGHT if heading_error > 0 else Command.LEFT
                print(f"[GOAL] Turn {direction.name}  {abs(heading_error):.1f}deg -> {rotations:.2f}rot")
                self._execute_turn(pose, rotations, direction)
                return direction

        dist_px = _distance_px(pose.px, GOAL_POSITION_PX)
        if dist_px > GOAL_THRESHOLD_PX:
            drive_px  = min(dist_px - GOAL_THRESHOLD_PX, MAX_DRIVE_PX)
            rotations = _px_to_rotations(drive_px)
            print(f"[GOAL] Drive {drive_px:.0f}px -> {rotations:.2f}rot")
            self._execute_drive(pose, rotations)
            return Command.FORWARD

        print("[GOAL] At goal -- releasing.")
        self._transition(State.RELEASE)
        return Command.STOP

    # --- State: RELEASE -------------------------------------------------------

    def _release_balls(self) -> Command:
        robot.release()
        self._transition(State.DONE)
        return Command.RELEASE

    # --- Movement helpers -----------------------------------------------------

    def _execute_drive(self, pose, rotations):
        self._cal.record_drive(pose.px, rotations)
        robot.drive(rotations)
        self._pose.invalidate()

    def _execute_turn(self, pose, rotations, direction):
        self._cal.record_turn(pose.angle, rotations)
        robot.turn(rotations, direction.name)
        self._pose.invalidate()

    def _heading_error_to(self, pose, target_cm):
        """Heading error using cm positions (for ball targets and goal)."""
        bearing = angle_to_target(pose.pos, target_cm)
        return angle_error(pose.angle, bearing)

    def _heading_error_to_px(self, pose, target_px):
        """Heading error using px positions (for avoid waypoints)."""
        bearing = angle_to_target(pose.px, target_px)
        return angle_error(pose.angle, bearing)

    def _transition(self, new_state):
        print(f"[FSM] {self.state.name} -> {new_state.name}")
        self.state = new_state


# --- Module-level helpers -----------------------------------------------------

def _distance_px(a, b):
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _px_to_rotations(drive_px):
    return drive_px / calibration_pixels.ratio


def _angle_to_rotations(heading_error):
    return abs(heading_error) / calibration_angle.ratio * TURN_DAMPING
