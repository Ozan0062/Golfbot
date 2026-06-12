"""
state_machine.py — GolfBot decision logic.

This file contains only state transitions and decisions.
All implementation details live in dedicated modules:

    pose_cache.py          ArUco timeout / pose freshness
    route_manager.py       TSP route, ball gathering, px lookup
    tsp_christofides.py    Christofides algorithm
    navigation.py          angle_to_target, angle_error
    ev3_controller.py      robot hardware commands
    commands.py            Command enum
"""

from enum import Enum, auto

import controller.ev3_controller as robot
from controller.calibration_manager import CalibrationManager
from controller.commands import Command
from controller.navigation import (
    angle_to_target,
    angle_error,
    cm_to_pixels,
    safe_approach_point,
)
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager, RouteTarget
from config import (
    DEGREES_PER_ROTATION,
    FIELD_HEIGHT_CM,
    FIELD_SAFETY_MARGIN_CM,
    FIELD_WIDTH_CM,
    GOAL_POSITION_CM,
    GOAL_POSITION_PX,
    PIXELS_PER_ROTATION,
    WARPED_HEIGHT,
    WARPED_WIDTH,
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ALIGN_THRESHOLD_DEG = 5    # degrees — turn if heading error exceeds this
BALL_THRESHOLD_PX   = 25    # pixels  — collect if closer than this
GOAL_THRESHOLD_PX   = 30    # pixels  — release when this close to goal
REVERSE_ROTATIONS   = 1.5   # motor rotations per reverse manoeuvre
MAX_DRIVE_PX        = 80    # pixels  — cap each drive step so we re-align mid-journey


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    SEEK           = auto()
    REVERSE_WHITE  = auto()
    REVERSE_ORANGE = auto()
    DRIVE_GOAL     = auto()
    RELEASE        = auto()
    DONE           = auto()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GolfBotController:

    def __init__(self):
        self.state           = State.SEEK
        self._pose           = PoseCache()
        self._route          = RouteManager()
        self._cal            = CalibrationManager()   # debug only — not used for commands
        self._reversed       = False
        self._locked_target  = None   # type: RouteTarget | None
        print(f"[FSM] Ready.  Goal: {GOAL_POSITION_CM}")
        print(f"[FSM] Hardcoded: {DEGREES_PER_ROTATION} deg/rot, {PIXELS_PER_ROTATION} px/rot")

    def update(self, world: dict) -> Command:
        pose = self._pose.update(world)
        if pose is None:
            print("[FSM] ArUco not detected — waiting.")
            return Command.STOP

        self._cal.consume(pose.px, pose.angle)   # prints [CAL] debug only

        if self.state == State.SEEK:
            return self._seek(pose, world)

        if self.state == State.REVERSE_WHITE:
            return self._reverse_white(world)

        if self.state == State.REVERSE_ORANGE:
            return self._reverse_orange(world)

        if self.state == State.DRIVE_GOAL:
            return self._drive_goal(pose)

        if self.state == State.RELEASE:
            robot.release()
            self._go(State.DONE)
            return Command.RELEASE

        return Command.STOP   # DONE

    # -------------------------------------------------------------------------
    # States
    # -------------------------------------------------------------------------

    def _seek(self, pose, world) -> Command:
        # Acquire target only when we don't have one.
        # Once locked, keep driving toward the stored position regardless of
        # whether YOLO still sees the ball — it often disappears when the robot
        # is right on top of it.
        if self._locked_target is None:
            self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        if self._locked_target is None:
            print("[FSM] No balls visible — reversing.")
            self._route.clear()
            self._reversed = False
            self._go(State.REVERSE_WHITE)
            return Command.STOP

        target = self._locked_target

        # Keep tracking distance to the locked ball position for diagnostics.
        # The ball hasn't moved; we just use the stored pixel coordinate.
        dist_px = _dist_px(pose.px, target.px)

        approach_cm = safe_approach_point(
            target.cm,
            FIELD_SAFETY_MARGIN_CM,
            FIELD_WIDTH_CM,
            FIELD_HEIGHT_CM,
        )
        approach_px = cm_to_pixels(
            approach_cm,
            WARPED_WIDTH,
            WARPED_HEIGHT,
            FIELD_WIDTH_CM,
            FIELD_HEIGHT_CM,
        )
        approach_dist_px = _dist_px(pose.px, approach_px)

        # Navigate to the safe point first. Once there, face the actual ball so
        # the claw points toward it without driving the robot into the corner.
        heading_target = approach_cm if approach_dist_px > BALL_THRESHOLD_PX else target.cm
        target_bearing = angle_to_target(pose.pos, heading_target)
        err = angle_error(pose.angle, target_bearing)
        print(f"[FSM] SEEK  angle={pose.angle:.1f}°  "
              f"target_bearing={target_bearing:.1f}°  "
              f"err={err:.1f}°  dist={dist_px:.0f}px  "
              f"approach_dist={approach_dist_px:.0f}px")

        if abs(err) > ALIGN_THRESHOLD_DEG:
            rotations = abs(err) / DEGREES_PER_ROTATION
            cmd = Command.RIGHT if err > 0 else Command.LEFT
            print(f"[FSM] Turn {cmd.name}  {abs(err):.1f}° → {rotations:.2f} rot")
            self._cal.record_turn(pose.angle, rotations)   # debug only
            robot.turn(rotations, cmd.name)
            self._pose.invalidate()
            return cmd

        if approach_dist_px > BALL_THRESHOLD_PX:
            drive_px = min(approach_dist_px - BALL_THRESHOLD_PX, MAX_DRIVE_PX)
            rotations = drive_px / PIXELS_PER_ROTATION
            print(f"[FSM] Drive  {drive_px:.0f}px "
                  f"(of {approach_dist_px:.0f}px) → {rotations:.2f} rot")
            self._cal.record_drive(pose.px, rotations)   # debug only
            robot.drive(rotations)
            self._pose.invalidate()
            return Command.FORWARD

        # Close enough — collect, then unlock so the next ball gets a fresh pick
        print(f"[FSM] Collecting at safe approach dist={approach_dist_px:.0f}px")
        self._locked_target = None
        self._route.advance()
        robot.collect()
        self._pose.invalidate()
        return Command.COLLECT

    def _reverse_white(self, world) -> Command:
        if not self._reversed:
            robot.reverse(REVERSE_ROTATIONS)
            self._pose.invalidate()
            self._reversed = True
            return Command.BACKWARD

        self._reversed = False
        if world.get("white_balls") or world.get("ob"):
            self._go(State.SEEK)
        else:
            self._go(State.REVERSE_ORANGE)
        return Command.STOP

    def _reverse_orange(self, world) -> Command:
        if not self._reversed:
            robot.reverse(REVERSE_ROTATIONS)
            self._pose.invalidate()
            self._reversed = True
            return Command.BACKWARD

        self._reversed = False
        if world.get("ob"):
            self._go(State.SEEK)
        else:
            self._go(State.DRIVE_GOAL)
        return Command.STOP

    def _drive_goal(self, pose) -> Command:
        err = angle_error(pose.angle, angle_to_target(pose.pos, GOAL_POSITION_CM))

        if abs(err) > ALIGN_THRESHOLD_DEG:
            rotations = abs(err) / DEGREES_PER_ROTATION
            cmd = Command.RIGHT if err > 0 else Command.LEFT
            print(f"[FSM] Goal-align {cmd.name}  {abs(err):.1f}° → {rotations:.2f} rot")
            self._cal.record_turn(pose.angle, rotations)   # debug only
            robot.turn(rotations, cmd.name)
            self._pose.invalidate()
            return cmd

        dist_px = _dist_px(pose.px, GOAL_POSITION_PX)
        if dist_px > GOAL_THRESHOLD_PX:
            drive_px = min(dist_px - GOAL_THRESHOLD_PX, MAX_DRIVE_PX)
            rotations = drive_px / PIXELS_PER_ROTATION
            print(f"[FSM] Goal-drive  {drive_px:.0f}px (of {dist_px:.0f}px) → {rotations:.2f} rot")
            self._cal.record_drive(pose.px, rotations)   # debug only
            robot.drive(rotations)
            self._pose.invalidate()
            return Command.FORWARD

        print("[FSM] At goal — releasing.")
        self._go(State.RELEASE)
        return Command.STOP

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _go(self, new_state: State):
        print(f"[FSM] {self.state.name} → {new_state.name}")
        self.state = new_state


def _dist_px(a, b) -> float:
    if a is None or b is None:
        return 0.0
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])
