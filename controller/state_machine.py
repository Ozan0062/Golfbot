"""state_machine.py — FSM for golf bot behavior."""

from enum import Enum, auto
import math
import time

import controller.ev3_controller as robot
from controller.calibration_manager import CalibrationManager
from controller.calibration_tracker import calibration_angle_left, calibration_angle_right, calibration_pixels
from controller.commands import Command
from controller.navigation import (
    angle_to_target, angle_error, path_is_clear, obstacle_waypoint,
    classify_zone, wall_approach_angle, staging_point,
)
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager, RouteTarget
from config import GOAL_POSITION_CM, GOAL_POSITION_PX, WARPED_WIDTH, WARPED_HEIGHT


# --- Thresholds --------------------------------------------------------------

ALIGN_THRESHOLD_DEG = 2    # Below this, we consider ourselves "aligned" and drive straight.
MIN_TURN_ROTATIONS = 0.25   # Ignore turns smaller than this.
TURN_DAMPING = 0.6          # Reduce turn commands to prevent oscillation when close.

COLLECT_RADIUS_PX = 90      # Accepted radius for collecting a ball.
GOAL_THRESHOLD_PX = 100      # Close enough to goal to stop driving and release balls.
REVERSE_ROTATIONS = 1.5     # How far to reverse when no balls are visible.
MAX_DRIVE_PX = 80           # Cap on drive distance per cycle to allow for course correction.

CROSS_CLEARANCE_PX = 70     # Min distance from cross before avoidance triggers.
AVOID_WAYPOINT_DIST_PX = CROSS_CLEARANCE_PX * 2   # Waypoint offset from cross.
AVOID_ARRIVE_PX = 15        # Close enough to waypoint to consider it reached.

WALL_APPROACH_DRIVE_PX = 40  # Max drive step when doing a wall/corner final approach (no turns).
WALL_MARGIN_PX = 120       # Ball this close to a wall triggers wall approach.
STAGING_DISTANCE_PX = 170    # How far back from the ball the staging point is.
                             # Must be >= WALL_MARGIN_PX / cos(45°) ≈ 170 so corner
                             # staging points land outside the margin on both axes.

# 3 staging points: 3×, 2×, and 1× staging distance from the ball.
CORNER_STAGE_DISTANCES_PX = (STAGING_DISTANCE_PX * 3, STAGING_DISTANCE_PX * 2, STAGING_DISTANCE_PX)
FIELD_EDGE_MARGIN_PX = 30    # Clamp corner waypoints this far from field edges.


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
        self._locked_target    = None   # type RouteTarget | None
        self._avoid_target     = None   # type tuple | None
        self._corner_waypoints = []     # remaining staged corner waypoints (AVOID chains through these)
        self._corner_approach_angle = None  # approach angle for current corner (deg), for debug
        self._is_wall_ball     = False  # True when current target is near a wall/corner
        self._strict_align     = False  # True when within collect radius — enforce 2° hard
        print(f"[FSM] Ready.  Goal={GOAL_POSITION_CM}  "
              f"cal=L{calibration_angle_left.ratio:.1f}/R{calibration_angle_right.ratio:.1f}deg/rot  "
              f"{calibration_pixels.ratio:.1f}px/rot")

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
        # Always re-evaluate: pick the closest ball from current position each SEEK.
        self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        if self._locked_target is None:
            print("[FSM] No balls visible -- reversing.")
            self._route.clear()
            self._corner_waypoints = []
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
        if zone == "corner":
            self._is_wall_ball = True
            angle = wall_approach_angle(walls)
            if angle is not None:
                waypoints = _corner_approach_waypoints(
                    pose.px, target.px, angle,
                    CORNER_STAGE_DISTANCES_PX,
                    WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
                )
                if waypoints:
                    self._avoid_target          = waypoints[0]
                    self._corner_waypoints      = waypoints[1:]
                    self._corner_approach_angle = angle
                    labels = "  →  ".join(f"({w[0]:.0f},{w[1]:.0f})px"
                                          for w in waypoints)
                    print(f"[SEEK] corner ball (walls={walls}) -- "
                          f"{len(waypoints)}-stage approach: {labels}")
                    self._transition(State.AVOID)
                    return Command.STOP
                # Already at final staging point — drive straight in, no turns
                print(f"[SEEK] Already at corner staging point for walls={walls} -- straight approach")
                self._transition(State.APPROACH)
                return Command.STOP

        elif zone == "wall":
            self._is_wall_ball = True
            angle = wall_approach_angle(walls)
            if angle is not None:
                waypoints = _corner_approach_waypoints(
                    pose.px, target.px, angle,
                    CORNER_STAGE_DISTANCES_PX,
                    WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
                )
                if waypoints:
                    self._avoid_target          = waypoints[0]
                    self._corner_waypoints      = waypoints[1:]
                    self._corner_approach_angle = angle
                    labels = "  →  ".join(f"({w[0]:.0f},{w[1]:.0f})px"
                                          for w in waypoints)
                    print(f"[SEEK] wall ball (walls={walls}) -- "
                          f"{len(waypoints)}-stage approach: {labels}")
                    self._transition(State.AVOID)
                    return Command.STOP
                print(f"[SEEK] Already at wall staging point for walls={walls} -- straight approach")
                self._transition(State.APPROACH)
                return Command.STOP
        else:
            self._is_wall_ball     = False
            self._corner_waypoints = []

        # Path is clear -- proceed to alignment
        self._transition(State.ALIGN)
        return Command.STOP

    # --- State: AVOID ---------------------------------------------------------
    # Drive to the avoid target waypoint. First turn to face it, then drive toward it.
    # After reaching the waypoint, clear it and return to SEEK to re-check the path from the new position.

    def _avoid(self, pose) -> Command:
        wp = self._avoid_target
        if wp is None:
            self._transition(State.SEEK)
            return Command.STOP

        # Check arrival FIRST — at close range heading error is unreliable
        dist_px = _distance_px(pose.px, wp)
        if dist_px <= AVOID_ARRIVE_PX:
            if self._corner_waypoints:
                next_wp = self._corner_waypoints.pop(0)
                if self._corner_approach_angle is not None:
                    deg_err = angle_error(pose.angle, self._corner_approach_angle)
                    print(f"[AVOID] Stage reached -- abs={pose.angle:.1f}°  "
                          f"target={self._corner_approach_angle:.1f}°  "
                          f"err={deg_err:+.1f}°  "
                          f"advancing to ({next_wp[0]:.0f},{next_wp[1]:.0f})px  "
                          f"({len(self._corner_waypoints)} remaining)")
                else:
                    print(f"[AVOID] Stage reached -- advancing to "
                          f"({next_wp[0]:.0f},{next_wp[1]:.0f})px  "
                          f"({len(self._corner_waypoints)} stage(s) remaining)")
                self._avoid_target = next_wp
                return Command.STOP
            # All corner waypoints consumed — go straight to APPROACH (no SEEK re-evaluation)
            if self._is_wall_ball:
                if self._corner_approach_angle is not None:
                    deg_err = angle_error(pose.angle, self._corner_approach_angle)
                    print(f"[AVOID] All stages done -- abs={pose.angle:.1f}°  "
                          f"target={self._corner_approach_angle:.1f}°  err={deg_err:+.1f}°  "
                          f"-- straight to APPROACH (no turns)")
                else:
                    print("[AVOID] All corner stages done -- straight to APPROACH (no turns)")
                self._avoid_target = None
                self._transition(State.APPROACH)
                return Command.STOP
            print("[AVOID] Waypoint reached -- returning to SEEK")
            self._avoid_target = None
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
        drive_px  = min(dist_px - AVOID_ARRIVE_PX, MAX_DRIVE_PX)
        rotations = _px_to_rotations(drive_px)
        print(f"[AVOID] Drive {drive_px:.0f}px -> {rotations:.2f}rot")
        self._execute_drive(pose, rotations)
        return Command.FORWARD

    # --- State: ALIGN ---------------------------------------------------------
    # Turn to face the locked target. Once heading error is within threshold,
    # transition to APPROACH. Each frame either turns or passes through.

    def _align(self, pose) -> Command:
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        heading_error = self._heading_error_to(pose, target.cm)
        print(f"[ALIGN] Heading_Error!!!!!={heading_error}")
        print(f"[ALIGN] heading={pose.angle:.1f}deg  error={heading_error:.1f}deg")

        if abs(heading_error) <= ALIGN_THRESHOLD_DEG:
            self._strict_align = False
            self._transition(State.APPROACH)
            return Command.STOP

        rotations = _angle_to_rotations(heading_error)
        if rotations < MIN_TURN_ROTATIONS and not self._strict_align:
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
            if self._is_wall_ball:
                # Wall/corner: trust the staging angle — no turns, just grab
                heading_error = self._heading_error_to(pose, target.cm)
                print(f"[APPROACH] Wall collect  heading_err={heading_error:+.1f}° -- grabbing ball")
            else:
                heading_error = self._heading_error_to(pose, target.cm)
                if abs(heading_error) > ALIGN_THRESHOLD_DEG:
                    print(f"[APPROACH] Within collect radius but heading error={heading_error:.1f}° "
                          f"> {ALIGN_THRESHOLD_DEG}° -- forcing strict re-align")
                    self._strict_align = True
                    self._transition(State.ALIGN)
                    return Command.STOP
            print(f"[APPROACH] Within collect radius -- grabbing ball")
            self._locked_target = None
            self._route.advance()
            robot.collect()
            self._pose.invalidate()

            # Wall ball: reverse back to safe distance before resuming
            if self._is_wall_ball:
                reverse_rot = _px_to_rotations(STAGING_DISTANCE_PX)
                print(f"[APPROACH] Wall ball -- reversing {reverse_rot:.2f}rot")
                robot.reverse(reverse_rot)
                self._pose.invalidate()
                self._is_wall_ball = False

            self._transition(State.SEEK)
            return Command.COLLECT

        # Drive forward
        if self._is_wall_ball:
            # Wall/corner: no turns, same as normal but capped at 40px steps
            drive_px = min(dist_px - COLLECT_RADIUS_PX, WALL_APPROACH_DRIVE_PX)
            rotations = _px_to_rotations(drive_px)
            print(f"[APPROACH] Wall drive {drive_px:.0f}px -> {rotations:.2f}rot  (no re-align)")
            self._execute_drive(pose, rotations)
            # Stay in APPROACH — no ALIGN loop
            return Command.FORWARD
        else:
            drive_px  = min(dist_px - COLLECT_RADIUS_PX, MAX_DRIVE_PX)
            rotations = _px_to_rotations(drive_px)
            print(f"[APPROACH] Drive {drive_px:.0f}px -> {rotations:.2f}rot")
            self._execute_drive(pose, rotations)
            self._transition(State.ALIGN)
            return Command.FORWARD

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
        robot.gate_open()
        time.sleep(3)
        robot.gate_close()
        self._transition(State.DONE)
        return Command.RELEASE

    # --- Movement helpers -----------------------------------------------------

    def _execute_drive(self, pose, rotations):
        self._cal.record_drive(pose.px, rotations)
        robot.drive(rotations)
        self._pose.invalidate()

    def _execute_turn(self, pose, rotations, direction):
        self._cal.record_turn(pose.angle, rotations, direction.name)
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

def _corner_approach_waypoints(robot_px, ball_px, approach_angle_deg,
                               stage_distances, field_w, field_h, margin):
    """
    Build an ordered list of waypoints along the approach axis for a corner ball.

    Stages are ordered far→close (e.g. 510px → 340px → 170px from the ball).
    A stage is included only if the robot is currently farther from the ball
    than that stage — i.e. we never route the robot backwards away from the ball.
    Waypoints are clamped to stay inside the field boundary.
    """
    robot_dist = _distance_px(robot_px, ball_px)
    waypoints = []
    for dist in stage_distances:          # already ordered far→close
        if robot_dist <= dist:
            continue                      # robot is already at or past this stage
        sp = staging_point(ball_px, approach_angle_deg, dist)
        sp = (
            max(margin, min(sp[0], field_w - margin)),
            max(margin, min(sp[1], field_h - margin)),
        )
        if _distance_px(robot_px, sp) > AVOID_ARRIVE_PX:
            waypoints.append(sp)
    return waypoints


def _distance_px(a, b):
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _px_to_rotations(drive_px):
    return drive_px / calibration_pixels.ratio


def _angle_to_rotations(heading_error):
    tracker = calibration_angle_right if heading_error > 0 else calibration_angle_left
    return abs(heading_error) / tracker.ratio * TURN_DAMPING
