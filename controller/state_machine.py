"""
state_machine.py — the GolfBot "brain".

Every camera frame, main.py calls `controller.update(world)`.  The controller
looks at where the robot and balls are, decides one thing to do, sends that
command to the EV3, and returns the Command so the overlay can show it.

It is a finite state machine.  Each state does ONE job and then either stays
put or hands off to the next state:

    ┌──────────────────────────  collect one ball  ──────────────────────────┐
    │                                                                         │
    SEEK ──► AVOID ──► ALIGN ──► APPROACH ──► (grab) ──► SEEK ◄────────────────┘
     │  pick a target   │  drive   │  turn to   │  drive to
     │                  │  around  │  face it   │  the ball
     │                  │  cross / │
     │                  │  to wall │
     │                  │  staging │
     │
     └─ no balls left ─► REVERSE_WHITE ─► REVERSE_ORANGE ─► DRIVE_GOAL ─► RELEASE ─► DONE
        (back up and rescan for balls)      (go dump the collected balls in the goal)

Tuning knobs live in the THRESHOLDS block below.  Everything is logged through
golfbot_logger: INFO = the story ("target locked", "collected ball"), DEBUG =
the per-frame numbers (turn/drive amounts, headings) which also go to the log
file.  Run with LOG_LEVEL=DEBUG to see the detail on the console.
"""

from enum import Enum, auto
import math
import time

import controller.ev3_controller as robot
from controller.calibration_manager import CalibrationManager
from controller.calibration_tracker import (
    calibration_angle_left, calibration_angle_right, calibration_pixels,
)
from controller.commands import Command
from controller.navigation import (
    angle_to_target, angle_error, path_is_clear, obstacle_waypoint,
    classify_zone, wall_approach_angle, staging_point,
)
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager
from config import GOAL_POSITION_CM, GOAL_POSITION_PX, WARPED_WIDTH, WARPED_HEIGHT
from golfbot_logger import get_logger

log = get_logger(__name__)


# --- Thresholds --------------------------------------------------------------

ALIGN_THRESHOLD_DEG = 2      # Below this heading error we count as "aligned" and drive straight.
MIN_TURN_ROTATIONS  = 0.25   # Ignore turns smaller than this.
TURN_DAMPING        = 0.6    # Scale turn commands down to prevent oscillation when close.

COLLECT_RADIUS_PX = 90       # Close enough to a ball to grab it.
GOAL_THRESHOLD_PX = 100      # Close enough to the goal to stop and release.
REVERSE_ROTATIONS = 1.5      # How far to back up when no balls are visible.
MAX_DRIVE_PX      = 80       # Cap on drive distance per cycle, so we re-check often.

CROSS_CLEARANCE_PX     = 70                    # Stay at least this far from the cross.
AVOID_WAYPOINT_DIST_PX = CROSS_CLEARANCE_PX * 2  # How far to the side the dodge waypoint sits.
AVOID_ARRIVE_PX        = 15                     # Close enough to a waypoint to count as reached.

WALL_MARGIN_PX      = 120    # A ball this close to a wall needs a staged approach.
STAGING_DISTANCE_PX = 170    # Standoff distance for the final straight-in approach.
                             # Must be >= WALL_MARGIN_PX / cos(45°) ≈ 170 so corner
                             # staging points land outside the margin on both axes.

# Two staging points per wall/corner ball: 2× then 1× the staging distance.
CORNER_STAGE_DISTANCES_PX = (STAGING_DISTANCE_PX * 2, STAGING_DISTANCE_PX)
FIELD_EDGE_MARGIN_PX = 30    # Keep staging waypoints this far inside the field edges.

_GOAL_APPROACH_ANGLE = 180.0  # Goal is on the left wall, so we approach heading left (180°).


# --- States ------------------------------------------------------------------

class State(Enum):
    SEEK           = auto()   # pick the next ball to go for
    AVOID          = auto()   # drive to a staging/dodge waypoint
    ALIGN          = auto()   # turn to face the locked target
    APPROACH       = auto()   # drive toward the locked target
    REVERSE_WHITE  = auto()   # back up, rescan for white (or orange) balls
    REVERSE_ORANGE = auto()   # back up, rescan for the orange ball only
    DRIVE_GOAL     = auto()   # navigate to the goal zone
    RELEASE        = auto()   # dump the balls at the goal
    DONE           = auto()   # mission complete


# --- Controller --------------------------------------------------------------

class GolfBotController:
    """Holds the current state plus the small amount of memory the FSM needs."""

    def __init__(self):
        self.state = State.SEEK
        self._pose  = PoseCache()
        self._route = RouteManager()
        self._cal   = CalibrationManager()

        self._locked_target        = None   # RouteTarget the robot is going for
        self._avoid_target         = None   # waypoint currently being driven to (px)
        self._corner_waypoints     = []     # staging waypoints still to visit after the current one
        self._corner_approach_angle = None  # heading to hold through a wall/corner approach (deg)
        self._goal_waypoints       = None   # None = not built yet; [] = staging done
        self._is_wall_ball         = False  # current target needs a staged wall/corner approach
        self._strict_align         = False  # enforce the 2° tolerance even for tiny turns
        self._has_reversed         = False  # already backed up this REVERSE cycle
        self._pose_ok              = True   # for logging pose-lost / reacquired only once

        log.info(
            "Controller ready — goal at %s cm | calibration: turn L %.1f / R %.1f deg-per-rot, drive %.1f px-per-rot",
            GOAL_POSITION_CM, calibration_angle_left.ratio,
            calibration_angle_right.ratio, calibration_pixels.ratio,
        )

    # -- Main entry point (called once per camera frame) ----------------------

    def update(self, world: dict) -> Command:
        """Run one tick of the state machine and return the command issued."""
        pose = self._pose.update(world)
        if pose is None:
            if self._pose_ok:
                log.warning("Robot pose lost (ArUco marker not detected) — holding")
                self._pose_ok = False
            return Command.STOP
        if not self._pose_ok:
            log.info("Robot pose reacquired")
            self._pose_ok = True

        self._cal.consume(pose.px, pose.angle)

        handler = self._HANDLERS.get(self.state)
        return handler(self, pose, world) if handler else Command.STOP

    # --- State: SEEK ---------------------------------------------------------

    def _seek(self, pose, world) -> Command:
        """Pick the closest remaining ball and decide how to approach it."""
        self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        if self._locked_target is None:
            log.info("No balls in view — backing up to rescan")
            self._reset_targeting()
            self._has_reversed = False
            self._transition(State.REVERSE_WHITE)
            return Command.STOP

        target = self._locked_target
        log.info("Target locked — ball at (%.0f, %.0f) px", target.px[0], target.px[1])

        # 1. Is the cross in the way?  Dodge around it first, then re-plan.
        if self._cross_blocks_path(pose, target, world):
            return Command.STOP

        # 2. Is the ball against a wall or in a corner?  Approach via staging points.
        zone, walls = classify_zone(target.px, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
        if zone in ("wall", "corner"):
            self._is_wall_ball = True
            self._transition(self._begin_staged_approach(pose, target, walls, zone))
            return Command.STOP

        # 3. Open field — just turn and drive.
        self._reset_targeting()
        self._is_wall_ball = False
        self._transition(State.ALIGN)
        return Command.STOP

    def _cross_blocks_path(self, pose, target, world) -> bool:
        """If the cross blocks the straight path, queue a dodge waypoint and enter AVOID."""
        cross_px = world.get("cross_px")
        if cross_px is None:
            return False

        clear, _ = path_is_clear(pose.px, target.px, [cross_px], CROSS_CLEARANCE_PX)
        if clear:
            return False

        wp = obstacle_waypoint(pose.px, target.px, cross_px,
                               AVOID_WAYPOINT_DIST_PX, WARPED_WIDTH, WARPED_HEIGHT)
        if wp is None:
            return False

        # A cross dodge is a single waypoint; after reaching it we re-plan from SEEK.
        self._avoid_target          = wp
        self._corner_waypoints      = []
        self._corner_approach_angle = None
        self._is_wall_ball          = False
        log.info("Cross blocks the path — steering around it")
        log.debug("dodge waypoint at (%.0f, %.0f) px", wp[0], wp[1])
        self._transition(State.AVOID)
        return True

    def _begin_staged_approach(self, pose, target, walls, zone) -> State:
        """Plan a wall/corner approach.  Returns the next state to enter."""
        angle = wall_approach_angle(walls)
        if angle is None:                       # no usable constraint — treat as open field
            return State.ALIGN

        waypoints = _corner_approach_waypoints(
            pose.px, target.px, angle, CORNER_STAGE_DISTANCES_PX,
            WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
        )
        if not waypoints:                       # already at the staging point — drive straight in
            log.debug("Already at %s staging point (walls=%s) — straight approach", zone, walls)
            return State.APPROACH

        self._avoid_target          = waypoints[0]
        self._corner_waypoints      = waypoints[1:]
        self._corner_approach_angle = angle
        log.info("%s ball — approaching via %d staging point(s)", zone.capitalize(), len(waypoints))
        log.debug("staging path: %s",
                  "  ->  ".join(f"({w[0]:.0f},{w[1]:.0f})" for w in waypoints))
        return State.AVOID

    # --- State: AVOID --------------------------------------------------------

    def _avoid(self, pose, world) -> Command:
        """Drive to the current waypoint; on arrival advance the plan."""
        wp = self._avoid_target
        if wp is None:
            self._transition(State.SEEK)
            return Command.STOP

        if _distance_px(pose.px, wp) <= AVOID_ARRIVE_PX:
            return self._waypoint_reached(pose)

        return self._drive_toward_waypoint(pose, wp)

    def _waypoint_reached(self, pose) -> Command:
        """Decide what to do once the robot reaches the current AVOID waypoint."""
        # More staging points queued → head to the next one.
        if self._corner_waypoints:
            self._avoid_target = self._corner_waypoints.pop(0)
            if self._corner_approach_angle is not None:
                log.debug("Stage reached (heading err %+.1f°) — advancing to (%.0f,%.0f), %d left",
                          angle_error(pose.angle, self._corner_approach_angle),
                          self._avoid_target[0], self._avoid_target[1], len(self._corner_waypoints))
            return Command.STOP

        # Staged wall/corner approach finished → aim and drive into the ball.
        if self._is_wall_ball:
            log.debug("Staging complete — aligning for final approach")
            self._corner_approach_angle = None
            self._avoid_target = None
            self._transition(State.ALIGN)
            return Command.STOP

        # Cross dodge finished → re-plan from the new position.
        log.debug("Reached dodge waypoint — re-checking the path")
        self._avoid_target = None
        self._transition(State.SEEK)
        return Command.STOP

    # --- State: ALIGN --------------------------------------------------------

    def _align(self, pose, world) -> Command:
        """Turn to face the locked target.  Once aligned, hand off to APPROACH."""
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        heading_error = self._heading_error_to(pose, target.cm)
        log.debug("ALIGN heading=%.1f°  error=%.1f°", pose.angle, heading_error)

        if abs(heading_error) <= ALIGN_THRESHOLD_DEG:
            self._strict_align = False
            self._transition(State.APPROACH)
            return Command.STOP

        rotations = _angle_to_rotations(heading_error)
        if rotations < MIN_TURN_ROTATIONS and not self._strict_align:
            self._transition(State.APPROACH)   # turn too small to bother — close enough
            return Command.STOP

        direction = Command.RIGHT if heading_error > 0 else Command.LEFT
        log.debug("ALIGN turn %s %.1f° -> %.2f rot", direction.name, abs(heading_error), rotations)
        self._execute_turn(pose, rotations, direction)
        return direction   # stay in ALIGN; next frame re-checks the heading

    # --- State: APPROACH -----------------------------------------------------

    def _approach(self, pose, world) -> Command:
        """Drive toward the locked target, re-aligning between steps."""
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        dist_px = _distance_px(pose.px, target.px)
        log.debug("APPROACH dist=%.0f px", dist_px)

        if dist_px > COLLECT_RADIUS_PX:
            drive_px = min(dist_px - COLLECT_RADIUS_PX, MAX_DRIVE_PX)
            self._execute_drive(pose, _px_to_rotations(drive_px))
            log.debug("APPROACH drive %.0f px", drive_px)
            self._transition(State.ALIGN)       # re-check heading before the next step
            return Command.FORWARD

        # Within collect radius — but make sure we are actually pointed at the ball.
        heading_error = self._heading_error_to(pose, target.cm)
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            log.debug("At the ball but off-heading (%.1f°) — strict re-align", heading_error)
            self._strict_align = True
            self._transition(State.ALIGN)
            return Command.STOP

        return self._grab_ball(pose)

    def _grab_ball(self, pose) -> Command:
        """Close the claw on the locked target and return to SEEK."""
        log.info("Collected ball")
        self._locked_target = None
        self._route.advance()
        robot.collect()
        self._pose.invalidate()

        if self._is_wall_ball:                  # back off so we don't shove the ball into the wall
            reverse_rot = _px_to_rotations(STAGING_DISTANCE_PX)
            log.debug("Wall ball — backing off %.2f rot", reverse_rot)
            robot.reverse(reverse_rot)
            self._pose.invalidate()
            self._is_wall_ball = False

        self._transition(State.SEEK)
        return Command.COLLECT

    # --- States: REVERSE_WHITE / REVERSE_ORANGE ------------------------------

    def _reverse_white(self, pose, world) -> Command:
        return self._handle_reverse(world, scan_for=("white_balls", "ob"),
                                    next_if_found=State.SEEK,
                                    next_if_empty=State.REVERSE_ORANGE)

    def _reverse_orange(self, pose, world) -> Command:
        return self._handle_reverse(world, scan_for=("ob",),
                                    next_if_found=State.SEEK,
                                    next_if_empty=State.DRIVE_GOAL)

    def _handle_reverse(self, world, scan_for, next_if_found, next_if_empty) -> Command:
        """Back up once, then on the next tick check whether anything appeared."""
        if not self._has_reversed:
            robot.reverse(REVERSE_ROTATIONS)
            self._pose.invalidate()
            self._has_reversed = True
            return Command.BACKWARD

        self._has_reversed = False
        found = any(world.get(key) for key in scan_for)
        log.debug("Rescan after reverse: %s", "found a ball" if found else "still empty")
        self._transition(next_if_found if found else next_if_empty)
        return Command.STOP

    # --- State: DRIVE_GOAL ---------------------------------------------------
    # Same staged approach as wall balls: drive the staging waypoints, then a
    # straight final drive in (no turns).  Goal is on the left wall (180°).

    def _drive_to_goal(self, pose, world) -> Command:
        if self._goal_waypoints is None:        # first entry — plan the staging path
            self._goal_waypoints = _corner_approach_waypoints(
                pose.px, GOAL_POSITION_PX, _GOAL_APPROACH_ANGLE,
                CORNER_STAGE_DISTANCES_PX, WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
            )
            log.info("Driving to goal — %d staging point(s)", len(self._goal_waypoints))

        # Work through the staging waypoints (turn + drive, like AVOID).
        if self._goal_waypoints:
            wp = self._goal_waypoints[0]
            if _distance_px(pose.px, wp) <= AVOID_ARRIVE_PX:
                self._goal_waypoints.pop(0)
                log.debug("Goal stage reached — %d left", len(self._goal_waypoints))
                return Command.STOP
            return self._drive_toward_waypoint(pose, wp)

        # Staging done — drive straight in until close enough to release.
        dist_px = _distance_px(pose.px, GOAL_POSITION_PX)
        if dist_px > GOAL_THRESHOLD_PX:
            drive_px = min(dist_px - GOAL_THRESHOLD_PX, MAX_DRIVE_PX)
            self._execute_drive(pose, _px_to_rotations(drive_px))
            log.debug("Goal final drive %.0f px", drive_px)
            return Command.FORWARD

        log.info("Reached goal — releasing balls")
        self._goal_waypoints = None             # reset in case of another run
        self._transition(State.RELEASE)
        return Command.STOP

    # --- State: RELEASE ------------------------------------------------------

    def _release_balls(self, pose, world) -> Command:
        robot.gate_open()
        time.sleep(3)
        robot.gate_close()
        log.info("Balls released — mission complete")
        self._transition(State.DONE)
        return Command.RELEASE

    def _done(self, pose, world) -> Command:
        return Command.STOP

    # --- Shared movement helpers ---------------------------------------------

    def _drive_toward_waypoint(self, pose, wp) -> Command:
        """Turn to face a pixel waypoint, or drive toward it if already aligned."""
        heading_error = self._heading_error_to_px(pose, wp)
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            rotations = _angle_to_rotations(heading_error)
            if rotations >= MIN_TURN_ROTATIONS:
                direction = Command.RIGHT if heading_error > 0 else Command.LEFT
                log.debug("waypoint turn %s %.1f° -> %.2f rot", direction.name,
                          abs(heading_error), rotations)
                self._execute_turn(pose, rotations, direction)
                return direction

        drive_px = min(_distance_px(pose.px, wp) - AVOID_ARRIVE_PX, MAX_DRIVE_PX)
        self._execute_drive(pose, _px_to_rotations(drive_px))
        log.debug("waypoint drive %.0f px", drive_px)
        return Command.FORWARD

    def _execute_drive(self, pose, rotations):
        self._cal.record_drive(pose.px, rotations)
        robot.drive(rotations)
        self._pose.invalidate()

    def _execute_turn(self, pose, rotations, direction):
        self._cal.record_turn(pose.angle, rotations, direction.name)
        robot.turn(rotations, direction.name)
        self._pose.invalidate()

    def _heading_error_to(self, pose, target_cm):
        """Heading error toward a cm target (balls and goal)."""
        return angle_error(pose.angle, angle_to_target(pose.pos, target_cm))

    def _heading_error_to_px(self, pose, target_px):
        """Heading error toward a pixel target (avoid/goal waypoints)."""
        return angle_error(pose.angle, angle_to_target(pose.px, target_px))

    def _reset_targeting(self):
        """Clear any staged-approach plan."""
        self._route.clear()
        self._corner_waypoints      = []
        self._corner_approach_angle = None

    def _transition(self, new_state):
        log.debug("%s -> %s", self.state.name, new_state.name)
        self.state = new_state

    # --- Read-only view for the camera overlay -------------------------------

    def debug_view(self) -> dict:
        """
        Snapshot of what the controller is doing, for draw_debug_overlay().

        Returns the locked target, the waypoint currently being driven to, and
        any waypoints queued after it (so the overlay can show "driving to" and
        "then").
        """
        if self.state == State.AVOID:
            current_wp = self._avoid_target
            upcoming   = list(self._corner_waypoints)
        elif self.state == State.DRIVE_GOAL and self._goal_waypoints:
            current_wp = self._goal_waypoints[0]
            upcoming   = list(self._goal_waypoints[1:])
        else:
            current_wp = None
            upcoming   = []

        return {
            "state":          self.state.name,
            "target":         self._locked_target,
            "avoid_target":   current_wp,
            "next_waypoints": upcoming,
        }

    # Dispatch table: state -> handler.  Defined last so the methods exist.
    _HANDLERS = {
        State.SEEK:           _seek,
        State.AVOID:          _avoid,
        State.ALIGN:          _align,
        State.APPROACH:       _approach,
        State.REVERSE_WHITE:  _reverse_white,
        State.REVERSE_ORANGE: _reverse_orange,
        State.DRIVE_GOAL:     _drive_to_goal,
        State.RELEASE:        _release_balls,
        State.DONE:           _done,
    }


# --- Module-level geometry helpers -------------------------------------------

def _corner_approach_waypoints(robot_px, ball_px, approach_angle_deg,
                               stage_distances, field_w, field_h, margin):
    """
    Ordered waypoints along the approach axis for a wall/corner (or goal) ball.

    Stages run far→close (e.g. 340px then 170px from the ball).  A stage is
    skipped only if the robot has already driven past it along the approach
    axis (measured by projecting the robot onto the approach direction, so a
    robot off to the side is never wrongly treated as "past" a stage).  Each
    waypoint is clamped to stay inside the field boundary.
    """
    angle_rad = math.radians(approach_angle_deg)
    # Unit vector pointing away from the ball along the approach axis
    # (the direction the robot comes in from).
    behind_x = -math.cos(angle_rad)
    behind_y = -math.sin(angle_rad)
    # How far "behind" the ball the robot currently is along that axis.
    robot_proj = ((robot_px[0] - ball_px[0]) * behind_x +
                  (robot_px[1] - ball_px[1]) * behind_y)

    waypoints = []
    for dist in stage_distances:                # already ordered far→close
        if robot_proj <= dist:
            continue                            # robot is nearer than this stage; skip it
        sp = staging_point(ball_px, approach_angle_deg, dist)
        sp = (
            max(margin, min(sp[0], field_w - margin)),
            max(margin, min(sp[1], field_h - margin)),
        )
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
