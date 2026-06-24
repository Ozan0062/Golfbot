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
import math
import time

import controller.ev3_controller as robot
from controller.calibration_manager import CalibrationManager
from controller.calibration_tracker import (
    calibration_angle_left, calibration_angle_right, calibration_pixels,
)
from controller.commands import Command
from controller.motion import (
    Driver, corner_approach_waypoints, px_to_rotations, angle_to_rotations,
)
from controller.navigation import (
    angle_to_target, angle_error, path_is_clear, obstacle_waypoint,
    classify_zone, wall_approach_angle, px_angle_to_cm,
    cross_approach_angle, cross_trigger_radius, cross_avoid_points,
)
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager
from controller.dijkstras import get_path
from config import (
    GOAL_POSITION_CM, GOAL_POSITION_PX, WARPED_WIDTH, WARPED_HEIGHT,
    FIELD_WIDTH_CM, FIELD_HEIGHT_CM,
    ALIGN_THRESHOLD_DEG, COLLECT_RADIUS_CM, REVERSE_ROTATIONS,
    CROSS_CLEARANCE_PX, AVOID_WAYPOINT_DIST_PX, AVOID_ARRIVE_PX,
    WALL_MARGIN_PX, CORNER_STAGE_DISTANCES_PX,
    FIELD_EDGE_MARGIN_PX, GOAL_APPROACH_ANGLE_DEG,
    MARKER_TO_CLAW_CM, CROSS_RADIUS_PX,
)
from golfbot_logger import get_logger
from vision.tracker import WorldState


# Smaller of the two anisotropic px/cm scales. Converting a claw-frame cm distance
# with the smaller scale gives a conservative pixel distance, so the claw always
# reaches the ball (never stalls short) whatever the heading.
_PX_PER_CM_MIN = min(WARPED_WIDTH / FIELD_WIDTH_CM, WARPED_HEIGHT / FIELD_HEIGHT_CM)

# On the final drive-in, stop the marker about one arm-length short of the ball so
# the claw lands on it. The precise grab is gated on the cm claw-to-ball distance
# (COLLECT_RADIUS_CM), not on this coarse pixel arrival.
_APPROACH_ARRIVE_PX = MARKER_TO_CLAW_CM * _PX_PER_CM_MIN


def _claw_tip_cm(robot_pos_cm, robot_angle_deg):
    """
    Claw-tip position on the floor, in cm: MARKER_TO_CLAW_CM forward of the
    height-corrected marker, along the heading. Exact for any heading because it
    is computed in cm, not in the anisotropic warped-pixel frame.
    """
    rad = math.radians(robot_angle_deg)
    return (robot_pos_cm[0] + MARKER_TO_CLAW_CM * math.cos(rad),
            robot_pos_cm[1] + MARKER_TO_CLAW_CM * math.sin(rad))



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

    def update(self, world:WorldState) -> Command:
        """Run one tick of the state machine and return the command issued."""
        # updates robot position
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
        
        if self._locked_target is None:
            path = get_path(world)
            self._locked_target = self._route.get_target_dijkstras(path, pose.px, world)
        
        if self._locked_target is None:
            log.info("No balls in view — backing up to rescan")
            self._reset_targeting()
            self._has_reversed = False
            self._transition(State.REVERSE_WHITE)
            return Command.STOP

        target = self._locked_target
        log.info("Target locked — ball at (%.0f, %.0f) px", target.px[0], target.px[1])

        # 1. Ball sitting in/at the cross?  Collect it like a corner ball
        #    (staged diagonal approach).  Must come before the dodge check below,
        #    otherwise the cross would read as "blocking" its own ball forever.
        if self._cross_ball_approach(pose, target, world):
            return Command.STOP

        # 2. Cross in the way of a different ball?  Dodge around it first, re-plan.
        if self._cross_blocks_path(pose, target, world):
            return Command.STOP

        # 3. Ball against a wall or in a corner?  Approach via a staging point.
        zone, walls = classify_zone(target.px, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
        if zone in ("wall", "corner"):
            self._is_wall_ball = True
            self._transition(self._begin_staged_approach(pose, target, walls, zone, world))
            return Command.STOP

        # 4. Open field — go straight to it.
        self._reset_targeting()
        self._is_wall_ball = False
        self._transition(State.APPROACH)
        return Command.STOP

    def _cross_blocks_path(self, pose, target, world) -> bool:
        """If the cross blocks the straight path, plan a full dodge route and enter AVOID."""
        cross_px = world.cross_px
        if cross_px is None:
            return False

        direct_clear, _ = path_is_clear(pose.px, target.px, [cross_px], CROSS_CLEARANCE_PX)
        if direct_clear:
            log.debug("Cross check: direct path (%.0f,%.0f)→(%.0f,%.0f) is clear",
                      pose.px[0], pose.px[1], target.px[0], target.px[1])
            return False

        log.info("Cross BLOCKS path — robot=(%.0f,%.0f) target=(%.0f,%.0f) cross=(%.0f,%.0f) clearance=%dpx",
                 pose.px[0], pose.px[1], target.px[0], target.px[1],
                 cross_px[0], cross_px[1], CROSS_CLEARANCE_PX)

        # Use the 4 fixed avoid points (midpoint between cross and each wall).
        avoid_map = cross_avoid_points(cross_px, WARPED_WIDTH, WARPED_HEIGHT)
        avoid_pts = list(avoid_map.values())
        label_map = {0: "right", 90: "down", 180: "left", 270: "up"}
        log.debug("Avoid points: %s",
                  "  ".join(f"{label_map[k]}=(%.0f,%.0f)" % v for k, v in avoid_map.items()))

        def is_clear(a, b):
            ok, _ = path_is_clear(a, b, [cross_px], CROSS_CLEARANCE_PX)
            return ok

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        # Try single-point routes: robot → pt → target.
        route = None
        for i, (k, pt) in enumerate(avoid_map.items()):
            r2pt   = is_clear(pose.px, pt)
            pt2tgt = is_clear(pt, target.px)
            log.debug("  %s (%.0f,%.0f): robot→pt=%s  pt→target=%s",
                      label_map[k], pt[0], pt[1],
                      "OK" if r2pt else "BLOCKED",
                      "OK" if pt2tgt else "BLOCKED")

        single = [pt for pt in avoid_pts if is_clear(pt, target.px)]
        single.sort(key=lambda pt: dist(pose.px, pt))
        for pt in single:
            if is_clear(pose.px, pt):
                route = [pt]
                log.info("Cross blocks path — single-point route via (%.0f,%.0f)", pt[0], pt[1])
                break

        # Try two-point routes: robot → pt1 → pt2 → target.
        if route is None:
            log.debug("No single-point route found — trying two-point routes")
            pairs = [
                (pt1, pt2)
                for pt1 in avoid_pts for pt2 in avoid_pts
                if pt1 is not pt2
                and is_clear(pose.px, pt1)
                and is_clear(pt1, pt2)
                and is_clear(pt2, target.px)
            ]
            if pairs:
                pt1, pt2 = min(pairs, key=lambda p: dist(pose.px, p[0]) + dist(p[0], p[1]))
                route = [pt1, pt2]
                log.info("Cross blocks path — two-point route (%.0f,%.0f)→(%.0f,%.0f)",
                         pt1[0], pt1[1], pt2[0], pt2[1])
            else:
                log.debug("  No valid two-point pairs found either")

        if route is None:
            log.warning("Cross blocks path but no avoid route found — proceeding anyway")
            return False

        self._avoid_target          = route[0]
        self._corner_waypoints      = route[1:]
        self._corner_approach_angle = None
        self._is_wall_ball          = False
        self._transition(State.AVOID)
        return True

    def _begin_staged_approach(self, pose, target, walls, zone, world=None) -> State:
        """Plan a wall/corner approach.  Returns the next state to enter."""
        angle = wall_approach_angle(walls)
        if angle is None:
            log.warning("No usable wall angle for zone=%s — skipping staging", zone)
            return State.APPROACH
        return self._stage_along_angle(pose, target, angle, zone, world)

    def _cross_ball_approach(self, pose, target, world) -> bool:
        """
        If the locked ball sits within the cross's pickup radius, collect it like
        a corner ball: staged diagonal approach + back-off.  Returns True if it
        took over (the caller should then return STOP).
        """
        cross_px = world.cross_px
        if cross_px is None:
            return False

        radius = cross_trigger_radius(CROSS_RADIUS_PX)
        dist = math.hypot(target.px[0] - cross_px[0], target.px[1] - cross_px[1])
        if dist > radius:
            return False

        angle = cross_approach_angle(target.px, cross_px)
        # Treat it as a wall ball so we back off after grabbing (don't shove it
        # into the cross).
        self._is_wall_ball = True
        dx = target.px[0] - cross_px[0]
        dy = target.px[1] - cross_px[1]
        log.info("Ball at the cross (%.0f px away, r=%.0f) — cross=(%.0f,%.0f) ball=(%.0f,%.0f) "
                 "dx=%.0f dy=%.0f → approach=%.0f°",
                 dist, radius, cross_px[0], cross_px[1], target.px[0], target.px[1],
                 dx, dy, angle)
        self._transition(self._stage_along_angle(pose, target, angle, "cross", world))
        return True

    def _stage_along_angle(self, pose, target, approach_angle, label, world=None) -> State:
        """
        Build the staging waypoints for a fixed-heading approach (wall, corner, or
        cross) and load them into the AVOID plan.  Returns the next state.

        If world is provided, checks whether the path to the first staging point
        is blocked by the cross and inserts a dodge waypoint if so.
        """
        waypoints = corner_approach_waypoints(
            pose.px, target.px, approach_angle, CORNER_STAGE_DISTANCES_PX,
            WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
        )
        self._avoid_target          = waypoints[0]
        self._corner_waypoints      = waypoints[1:]
        # Staging points are placed in pixel space (raw `approach_angle`); the
        # heading we align to before driving in is the same approach in the cm
        # frame, so the two agree once pose.angle is physical. Axis-aligned
        # headings are unchanged.
        self._corner_approach_angle = px_angle_to_cm(approach_angle)
        log.info("%s ball — approaching via %d staging point(s)", label.capitalize(), len(waypoints))
        log.debug("staging path: %s",
                  "  ->  ".join(f"({w[0]:.0f},{w[1]:.0f})" for w in waypoints))

        # If the cross blocks the path to the first staging point, route around it
        # using the 4 fixed cardinal avoid points (same logic as _cross_blocks_path).
        cross_px = world.cross_px if world is not None else None
        if cross_px is not None:
            staging_pt = self._avoid_target
            staging_clear, _ = path_is_clear(pose.px, staging_pt, [cross_px], CROSS_CLEARANCE_PX)
            log.debug("Stage cross check: robot=(%.0f,%.0f) → staging=(%.0f,%.0f) = %s",
                      pose.px[0], pose.px[1], staging_pt[0], staging_pt[1],
                      "clear" if staging_clear else "BLOCKED")
            if not staging_clear:
                avoid_map = cross_avoid_points(cross_px, WARPED_WIDTH, WARPED_HEIGHT)
                avoid_pts = list(avoid_map.values())

                def _clr(a, b):
                    ok, _ = path_is_clear(a, b, [cross_px], CROSS_CLEARANCE_PX)
                    return ok

                def _dst(a, b):
                    return math.hypot(a[0] - b[0], a[1] - b[1])

                # Try single avoid point: robot → pt → staging.
                route = None
                single = [pt for pt in avoid_pts if _clr(pt, staging_pt)]
                single.sort(key=lambda pt: _dst(pose.px, pt))
                for pt in single:
                    if _clr(pose.px, pt):
                        route = [pt]
                        log.info("Cross blocks staging — single avoid point (%.0f,%.0f) → staging=(%.0f,%.0f)",
                                 pt[0], pt[1], staging_pt[0], staging_pt[1])
                        break

                # Try two avoid points: robot → pt1 → pt2 → staging.
                if route is None:
                    pairs = [
                        (pt1, pt2)
                        for pt1 in avoid_pts for pt2 in avoid_pts
                        if pt1 is not pt2
                        and _clr(pose.px, pt1)
                        and _clr(pt1, pt2)
                        and _clr(pt2, staging_pt)
                    ]
                    if pairs:
                        pt1, pt2 = min(pairs, key=lambda p: _dst(pose.px, p[0]) + _dst(p[0], p[1]))
                        route = [pt1, pt2]
                        log.info("Cross blocks staging — two avoid points (%.0f,%.0f)→(%.0f,%.0f) → staging=(%.0f,%.0f)",
                                 pt1[0], pt1[1], pt2[0], pt2[1], staging_pt[0], staging_pt[1])

                if route is not None:
                    self._corner_waypoints.insert(0, staging_pt)
                    self._avoid_target = route[0]
                    for extra in reversed(route[1:]):
                        self._corner_waypoints.insert(0, extra)
                else:
                    log.warning("Cross blocks staging but no avoid route found — proceeding anyway")

        return State.AVOID

    # --- State: AVOID --------------------------------------------------------

    def _avoid(self, pose, world) -> Command:
        """Drive to the current waypoint; on arrival advance the plan."""
        wp = self._avoid_target
        if wp is None:
            log.warning("AVOID entered with no waypoint — falling back to SEEK")
            self._transition(State.SEEK)
            return Command.STOP

        command, arrived = self._driver.drive_toward(pose, wp, AVOID_ARRIVE_PX)
        if arrived:
            log.info("Waypoint reached at (%.0f, %.0f)", wp[0], wp[1])
            return self._waypoint_reached(pose)
        return command

    def _waypoint_reached(self, pose) -> Command:
        """Decide what to do once the robot reaches the current AVOID waypoint."""
        # More staging points queued → head to the next one.
        if self._corner_waypoints:
            self._avoid_target = self._corner_waypoints.pop(0)
            log.info("Advancing to next waypoint (%.0f, %.0f), %d remaining",
                     self._avoid_target[0], self._avoid_target[1], len(self._corner_waypoints))
            return Command.STOP

        # Staged approach finished → align to the approach heading, then head straight in.
        if self._is_wall_ball:
            if self._corner_approach_angle is not None:
                heading_err = angle_error(pose.angle, self._corner_approach_angle)
                if abs(heading_err) > ALIGN_THRESHOLD_DEG:
                    direction = Command.RIGHT if heading_err > 0 else Command.LEFT
                    rotations = angle_to_rotations(heading_err)
                    log.debug("Pre-approach align %.1f° %s", abs(heading_err), direction.name)
                    self._driver.turn(pose, rotations, direction)
            log.debug("Staging complete — heading in to the ball")
            self._corner_approach_angle = None
            self._avoid_target = None
            log.info("Staging complete — transitioning to APPROACH")
            self._transition(State.APPROACH)
            return Command.STOP

        # Cross dodge finished → re-plan from the new position.
        log.info("Reached dodge waypoint — re-checking the path")
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

        # Arrival is gated on the claw tip (not the marker), measured in cm on the
        # floor plane where the scale is uniform: project MARKER_TO_CLAW_CM forward
        # of the height-corrected marker along the heading.
        claw = _claw_tip_cm(pose.pos, pose.angle)
        off  = math.hypot(claw[0] - target.cm[0], claw[1] - target.cm[1])

        # Not within collect range yet: drive the marker in (it stops ~one arm-length
        # short so the claw lands on the ball). If the marker reached that stop point
        # but the claw is still short, re-align when the heading swung it off,
        # otherwise nudge straight in by the shortfall. We never grab until the claw
        # is actually within COLLECT_RADIUS_CM.
        if off > COLLECT_RADIUS_CM:
            command, arrived = self._driver.drive_toward(pose, target.px, _APPROACH_ARRIVE_PX)
            if not arrived:
                return command
            heading_error = angle_error(pose.angle, angle_to_target(pose.pos, target.cm))
            if abs(heading_error) > ALIGN_THRESHOLD_DEG:
                direction = Command.RIGHT if heading_error > 0 else Command.LEFT
                log.debug("Claw short (%.1f cm) and off-heading (%.1f°) — turning %s",
                          off, heading_error, direction.name)
                self._driver.turn(pose, angle_to_rotations(heading_error), direction)
                return direction
            nudge_px = (off - COLLECT_RADIUS_CM) * _PX_PER_CM_MIN
            log.debug("Claw short by %.1f cm — nudging in %.0f px", off - COLLECT_RADIUS_CM, nudge_px)
            self._driver.drive(pose, px_to_rotations(nudge_px))
            return Command.FORWARD

        # Claw within collect range — check we're actually pointed at the ball.
        # Use the marker (rotation centre) for the bearing, not the claw tip.
        # The claw tip sweeps an arc during in-place turns, which shifts the
        # computed bearing on every frame and causes oscillation.
        heading_error = angle_error(pose.angle, angle_to_target(pose.pos, target.cm))
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            direction = Command.RIGHT if heading_error > 0 else Command.LEFT
            log.debug("At the ball but off-heading (%.1f°) — turning %s", heading_error, direction.name)
            self._driver.turn(pose, angle_to_rotations(heading_error), direction)
            return direction

        return self._grab_ball(pose)

    def _grab_ball(self, pose) -> Command:
        """Close the claw on the locked target and return to SEEK."""
        target  = self._locked_target
        claw    = _claw_tip_cm(pose.pos, pose.angle)
        dx      = claw[0] - target.cm[0]
        dy      = claw[1] - target.cm[1]
        deg_off = angle_error(pose.angle, angle_to_target(pose.pos, target.cm))
        log.debug(
            "Collecting — claw=(%.0f,%.0f) ball=(%.0f,%.0f) Δx=%.1f Δy=%.1f cm, %.1f° off",
            claw[0], claw[1], target.cm[0], target.cm[1], dx, dy, deg_off,
        )
        log.info("Collected ball")
        self._locked_target = None
        self._route.advance()
        robot.close_claw()
        self._pose.invalidate()

        if self._is_wall_ball:                  # back off so we don't shove the ball into the wall
            log.info("Wall/corner ball — backing off")
            self._driver.reverse(px_to_rotations(REVERSE_ROTATIONS))
            self._is_wall_ball = False
        
        robot.gate_rotate()       
        robot.reset_claw()

        self._transition(State.SEEK)
        return Command.COLLECT

    # --- States: REVERSE_WHITE / REVERSE_ORANGE ------------------------------

    def _reverse_white(self, pose, world) -> Command:
        return self._handle_reverse(world,
                                    scan_for=("white_balls", "white_wall_balls",
                                              "white_corner_balls", "ob"),
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
        found = any(getattr(world, key) for key in scan_for)
        log.debug("Rescan after reverse: %s", "found a ball" if found else "still empty")
        self._transition(next_if_found if found else next_if_empty)
        return Command.STOP

    # --- State: DRIVE_GOAL ---------------------------------------------------
    # Treat the goal like a wall ball: stage the approach, then drive in with
    # the same claw-tip logic as APPROACH. When the claw is within COLLECT_RADIUS_CM
    # of GOAL_POSITION_CM, release instead of grabbing.

    def _drive_to_goal(self, pose, world) -> Command:
        if self._goal_waypoints is None:        # first entry — build staging route
            self._goal_waypoints = corner_approach_waypoints(
                pose.px, GOAL_POSITION_PX, GOAL_APPROACH_ANGLE_DEG,
                CORNER_STAGE_DISTANCES_PX, WARPED_WIDTH, WARPED_HEIGHT, FIELD_EDGE_MARGIN_PX,
            )
            # Cross avoidance: if the cross blocks the first staging point, prepend avoid points.
            cross_px = world.cross_px
            if cross_px is not None and self._goal_waypoints:
                staging_pt = self._goal_waypoints[0]
                staging_clear, _ = path_is_clear(pose.px, staging_pt, [cross_px], CROSS_CLEARANCE_PX)
                if not staging_clear:
                    avoid_map = cross_avoid_points(cross_px, WARPED_WIDTH, WARPED_HEIGHT)
                    avoid_pts = list(avoid_map.values())

                    def _clr(a, b):
                        ok, _ = path_is_clear(a, b, [cross_px], CROSS_CLEARANCE_PX)
                        return ok

                    single = sorted([pt for pt in avoid_pts if _clr(pose.px, pt) and _clr(pt, staging_pt)],
                                    key=lambda p: math.hypot(p[0] - pose.px[0], p[1] - pose.px[1]))
                    if single:
                        self._goal_waypoints.insert(0, single[0])
                        log.info("Goal: cross blocks staging — avoid via (%.0f,%.0f)", single[0][0], single[0][1])
                    else:
                        pairs = [(p1, p2) for p1 in avoid_pts for p2 in avoid_pts
                                 if p1 is not p2 and _clr(pose.px, p1) and _clr(p1, p2) and _clr(p2, staging_pt)]
                        if pairs:
                            p1, p2 = min(pairs, key=lambda p: math.hypot(p[0][0]-pose.px[0], p[0][1]-pose.px[1]))
                            self._goal_waypoints[:0] = [p1, p2]
                            log.info("Goal: cross blocks staging — avoid via (%.0f,%.0f)→(%.0f,%.0f)",
                                     p1[0], p1[1], p2[0], p2[1])
                        else:
                            log.warning("Goal: cross blocks staging but no avoid route found — proceeding anyway")
            log.info("Driving to goal — %d staging waypoint(s)", len(self._goal_waypoints))

        # Phase 1: work through staging waypoints (same as AVOID).
        if self._goal_waypoints:
            command, arrived = self._driver.drive_toward(pose, self._goal_waypoints[0], AVOID_ARRIVE_PX)
            if arrived:
                self._goal_waypoints.pop(0)
                return Command.STOP
            return command

        # Phase 2: final approach — same logic as APPROACH, targeting GOAL_POSITION_CM.
        claw = _claw_tip_cm(pose.pos, pose.angle)
        off  = math.hypot(claw[0] - GOAL_POSITION_CM[0], claw[1] - GOAL_POSITION_CM[1])

        if off > COLLECT_RADIUS_CM:
            command, arrived = self._driver.drive_toward(pose, GOAL_POSITION_PX, _APPROACH_ARRIVE_PX)
            if not arrived:
                return command
            heading_error = angle_error(pose.angle, angle_to_target(pose.pos, GOAL_POSITION_CM))
            if abs(heading_error) > ALIGN_THRESHOLD_DEG:
                direction = Command.RIGHT if heading_error > 0 else Command.LEFT
                self._driver.turn(pose, angle_to_rotations(heading_error), direction)
                return direction
            nudge_px = (off - COLLECT_RADIUS_CM) * _PX_PER_CM_MIN
            log.debug("Goal: claw short by %.1f cm — nudging in %.0f px", off - COLLECT_RADIUS_CM, nudge_px)
            self._driver.drive(pose, px_to_rotations(nudge_px))
            return Command.FORWARD

        # Claw within range — align heading then release.
        heading_error = angle_error(pose.angle, angle_to_target(pose.pos, GOAL_POSITION_CM))
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            direction = Command.RIGHT if heading_error > 0 else Command.LEFT
            self._driver.turn(pose, angle_to_rotations(heading_error), direction)
            return direction

        log.info("Claw at goal (%.1f cm off) — releasing", off)
        self._goal_waypoints = None
        self._transition(State.RELEASE)
        return Command.STOP

    # --- State: RELEASE / DONE -----------------------------------------------

    def _release_balls(self, pose, world) -> Command:
        log.info("Releasing — robot at (%.1f, %.1f) cm, heading %.1f°",
                 pose.pos[0], pose.pos[1], pose.angle)
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
