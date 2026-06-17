"""
state_machine.py — the GolfBot "brain".

Every camera frame, main.py calls `controller.update(world)`.  The controller
decides ONE thing to do, the Driver carries it out, and the Command is returned
so the overlay can show it.

    ┌─────────────────────  collect one ball  ─────────────────────┐
    │                                                               │
    SEEK ──► AVOID ──► APPROACH ──► (grab) ──► SEEK ◄────────────────┘
     │  pick a    │ drive to     │ turn-to-face then drive in;
     │  target    │ staging /    │ grab when within reach
     │            │ around cross │
     │
     └─ no balls left ─► REVERSE_WHITE ─► REVERSE_ORANGE ─► DRIVE_GOAL ─► RELEASE ─► DONE

This file is the decision logic only.  The "how to move" details live in
controller/motion.py (the Driver + drive_toward primitive), and the tuning
knobs live in config.py.  Logging: INFO = the story, DEBUG = the per-frame
numbers (also written to the log file; run with LOG_LEVEL=DEBUG to see them).
"""

from enum import Enum, auto
import time

import controller.ev3_controller as robot
from controller.calibration_manager import CalibrationManager
from controller.calibration_tracker import (
    calibration_angle_left, calibration_angle_right, calibration_pixels,
)
from controller.commands import Command
from controller.motion import (
    Driver, corner_approach_waypoints, distance_px, px_to_rotations, angle_to_rotations,
)
from controller.navigation import (
    angle_to_target, angle_error, path_is_clear, obstacle_waypoint,
    classify_zone, wall_approach_angle,
)
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager
from config import (
    GOAL_POSITION_CM, GOAL_POSITION_PX, WARPED_WIDTH, WARPED_HEIGHT,
    ALIGN_THRESHOLD_DEG, COLLECT_RADIUS_PX, GOAL_THRESHOLD_PX, REVERSE_ROTATIONS,
    MAX_DRIVE_PX, CROSS_CLEARANCE_PX, AVOID_WAYPOINT_DIST_PX, AVOID_ARRIVE_PX,
    WALL_MARGIN_PX, STAGING_DISTANCE_PX, CORNER_STAGE_DISTANCES_PX,
    FIELD_EDGE_MARGIN_PX, GOAL_APPROACH_ANGLE_DEG,
)
from golfbot_logger import get_logger

log = get_logger(__name__)


class State(Enum):
    SEEK           = auto()   # pick the next ball to go for
    AVOID          = auto()   # drive to a staging/dodge waypoint
    APPROACH       = auto()   # turn to face the target, drive in, and grab it
    REVERSE_WHITE  = auto()   # back up, rescan for white (or orange) balls
    REVERSE_ORANGE = auto()   # back up, rescan for the orange ball only
    DRIVE_GOAL     = auto()   # navigate to the goal zone
    RELEASE        = auto()   # dump the balls at the goal
    DONE           = auto()   # mission complete


class GolfBotController:
    """Holds the current state plus the small amount of memory the FSM needs."""

    def __init__(self):
        self.state  = State.SEEK
        self._pose  = PoseCache()
        self._route = RouteManager()
        self._cal   = CalibrationManager()
        self._driver = Driver(self._cal, self._pose)

        self._locked_target         = None   # RouteTarget the robot is going for
        self._avoid_target          = None   # waypoint currently being driven to (px)
        self._corner_waypoints      = []     # staging waypoints still to visit
        self._corner_approach_angle = None   # heading held through a wall/corner approach (deg)
        self._goal_waypoints        = None   # None = not built yet; [] = staging done
        self._is_wall_ball          = False  # current target needs a staged approach
        self._has_reversed          = False  # already backed up this REVERSE cycle
        self._pose_ok               = True   # for logging pose-lost / reacquired once

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

        # 1. Cross in the way?  Dodge around it first, then re-plan.
        if self._cross_blocks_path(pose, target, world):
            return Command.STOP

        # 2. Ball against a wall or in a corner?  Approach via staging points.
        zone, walls = classify_zone(target.px, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
        if zone in ("wall", "corner"):
            self._is_wall_ball = True
            self._transition(self._begin_staged_approach(pose, target, walls, zone))
            return Command.STOP

        # 3. Open field — just go to it.
        self._reset_targeting()
        self._is_wall_ball = False
        self._transition(State.APPROACH)
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
            return State.APPROACH

        waypoints = corner_approach_waypoints(
            pose.px, target.px, angle, CORNER_STAGE_DISTANCES_PX,
            WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
        )
        if not waypoints:                       # already at the staging point — go straight in
            log.debug("Already at %s staging point (walls=%s) — heading straight in", zone, walls)
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

        command, arrived = self._driver.drive_toward(pose, wp, AVOID_ARRIVE_PX)
        if arrived:
            return self._waypoint_reached(pose)
        return command

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

        # Staged approach finished → head straight in to the ball.
        if self._is_wall_ball:
            log.debug("Staging complete — heading in to the ball")
            self._corner_approach_angle = None
            self._avoid_target = None
            self._transition(State.APPROACH)
            return Command.STOP

        # Cross dodge finished → re-plan from the new position.
        log.debug("Reached dodge waypoint — re-checking the path")
        self._avoid_target = None
        self._transition(State.SEEK)
        return Command.STOP

    # --- State: APPROACH -----------------------------------------------------
    # Turn to face the locked ball and drive toward it (one step per frame).
    # When within collect range, make sure we're pointed at it, then grab.

    def _approach(self, pose, world) -> Command:
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        command, arrived = self._driver.drive_toward(pose, target.px, COLLECT_RADIUS_PX)
        if not arrived:
            return command

        # Within reach — turn in place if we're not actually pointed at the ball.
        heading_error = angle_error(pose.angle, angle_to_target(pose.px, target.px))
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            direction = Command.RIGHT if heading_error > 0 else Command.LEFT
            log.debug("At the ball but off-heading (%.1f°) — turning %s", heading_error, direction.name)
            self._driver.turn(pose, angle_to_rotations(heading_error), direction)
            return direction

        return self._grab_ball(pose)

    def _grab_ball(self, pose) -> Command:
        """Close the claw on the locked target and return to SEEK."""
        log.info("Collected ball")
        self._locked_target = None
        self._route.advance()
        robot.collect()
        self._pose.invalidate()

        if self._is_wall_ball:                  # back off so we don't shove the ball into the wall
            log.debug("Wall ball — backing off")
            self._driver.reverse(px_to_rotations(STAGING_DISTANCE_PX))
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
            self._driver.reverse(REVERSE_ROTATIONS)
            self._has_reversed = True
            return Command.BACKWARD

        self._has_reversed = False
        found = any(world.get(key) for key in scan_for)
        log.debug("Rescan after reverse: %s", "found a ball" if found else "still empty")
        self._transition(next_if_found if found else next_if_empty)
        return Command.STOP

    # --- State: DRIVE_GOAL ---------------------------------------------------
    # Staged approach (like wall balls), then a straight final drive in (no turns).

    def _drive_to_goal(self, pose, world) -> Command:
        if self._goal_waypoints is None:        # first entry — plan the staging path
            self._goal_waypoints = corner_approach_waypoints(
                pose.px, GOAL_POSITION_PX, GOAL_APPROACH_ANGLE_DEG,
                CORNER_STAGE_DISTANCES_PX, WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
            )
            log.info("Driving to goal — %d staging point(s)", len(self._goal_waypoints))

        # Work through the staging waypoints (turn + drive, like AVOID).
        if self._goal_waypoints:
            command, arrived = self._driver.drive_toward(pose, self._goal_waypoints[0], AVOID_ARRIVE_PX)
            if arrived:
                self._goal_waypoints.pop(0)
                log.debug("Goal stage reached — %d left", len(self._goal_waypoints))
                return Command.STOP
            return command

        # Staging done — drive straight in (no turns) until close enough to release.
        dist = distance_px(pose.px, GOAL_POSITION_PX)
        if dist > GOAL_THRESHOLD_PX:
            drive_px = min(dist - GOAL_THRESHOLD_PX, MAX_DRIVE_PX)
            self._driver.drive(pose, px_to_rotations(drive_px))
            log.debug("Goal final drive %.0f px", drive_px)
            return Command.FORWARD

        log.info("Reached goal — releasing balls")
        self._goal_waypoints = None             # reset in case of another run
        self._transition(State.RELEASE)
        return Command.STOP

    # --- State: RELEASE / DONE -----------------------------------------------

    def _release_balls(self, pose, world) -> Command:
        robot.gate_open()
        time.sleep(3)
        robot.gate_close()
        log.info("Balls released — mission complete")
        self._transition(State.DONE)
        return Command.RELEASE

    def _done(self, pose, world) -> Command:
        return Command.STOP

    # --- Small helpers -------------------------------------------------------

    def _reset_targeting(self):
        """Clear any staged-approach plan."""
        self._route.clear()
        self._corner_waypoints      = []
        self._corner_approach_angle = None

    def _transition(self, new_state):
        log.debug("%s -> %s", self.state.name, new_state.name)
        self.state = new_state

    def debug_view(self) -> dict:
        """
        Snapshot of what the controller is doing, for draw_debug_overlay():
        the locked target, the waypoint being driven to now, and the ones after.
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
        State.APPROACH:       _approach,
        State.REVERSE_WHITE:  _reverse_white,
        State.REVERSE_ORANGE: _reverse_orange,
        State.DRIVE_GOAL:     _drive_to_goal,
        State.RELEASE:        _release_balls,
        State.DONE:           _done,
    }
