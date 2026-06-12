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
from controller.calibration_tracker import calibration_angle, calibration_pixels
from controller.commands import Command
from controller.navigation import angle_to_target, angle_error
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager, RouteTarget
from config import GOAL_POSITION_CM, GOAL_POSITION_PX


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ALIGN_THRESHOLD_DEG = 10    # degrees — turn if heading error exceeds this
MIN_TURN_ROTATIONS  = 0.15  # skip turns smaller than this — below motor's reliable resolution
BALL_THRESHOLD_PX   = 15    # pixels  — collect if closer than this
CLOSE_APPROACH_PX   = 60    # pixels  — inside this radius stop turning, just drive straight
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
        self._cal            = CalibrationManager()
        self._reversed       = False
        self._locked_target  = None   # type: RouteTarget | None
        print(f"[FSM] Ready.  Goal: {GOAL_POSITION_CM}")
        print(f"[FSM] Calibration init: {calibration_angle.ratio:.2f} deg/rot, {calibration_pixels.ratio:.2f} px/rot")

    def update(self, world: dict) -> Command:
        pose = self._pose.update(world)
        if pose is None:
            print("[FSM] ArUco not detected — waiting.")
            return Command.STOP

        self._cal.consume(pose.px, pose.angle)

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
            print("setting target")
            self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        if self._locked_target is None:
            print("[FSM] No balls visible — reversing.")
            self._route.clear()
            self._reversed = False
            self._go(State.REVERSE_WHITE)
            return Command.STOP

        target = self._locked_target

        dist_px      = _dist_px(pose.px, target.px)
        target_bearing = angle_to_target(pose.pos, target.cm)
        err          = angle_error(pose.angle, target_bearing)
        print(f"[FSM] SEEK  angle={pose.angle:.1f}°  "
              f"target_bearing={target_bearing:.1f}°  "
              f"err={err:.1f}°  dist={dist_px:.0f}px")

        # Collect first — never turn when already on top of the ball
        if dist_px <= BALL_THRESHOLD_PX:
            print(f"[FSM] Collecting at dist={dist_px:.0f}px")
            self._locked_target = None
            self._route.advance()
            robot.collect()
            self._pose.invalidate()
            return Command.COLLECT

        # Close approach — stop turning, drive straight.
        # ArUco jitter at close range (~6cm) swings the bearing 30-45°, so
        # alignment attempts cause oscillation instead of progress.
        if dist_px <= CLOSE_APPROACH_PX:
            drive_px = min(dist_px - BALL_THRESHOLD_PX, MAX_DRIVE_PX)
            rotations = drive_px / calibration_pixels.ratio
            print(f"[FSM] Close-drive  {drive_px:.0f}px (of {dist_px:.0f}px, err={err:.1f}°) → {rotations:.2f} rot")
            self._cal.record_drive(pose.px, rotations)
            robot.drive(rotations)
            self._pose.invalidate()
            return Command.FORWARD

        # Normal approach — align then drive
        if abs(err) > ALIGN_THRESHOLD_DEG:
            rotations = abs(err) / calibration_angle.ratio
            if rotations < MIN_TURN_ROTATIONS:
                print(f"[FSM] Turn skipped — {rotations:.2f} rot too small to be reliable")
            else:
                cmd = Command.RIGHT if err > 0 else Command.LEFT
                print(f"[FSM] Turn {cmd.name}  {abs(err):.1f}° → {rotations:.2f} rot  ({calibration_angle.ratio:.2f} deg/rot)")
                self._cal.record_turn(pose.angle, rotations)
                robot.turn(rotations, cmd.name)
                self._pose.invalidate()
                return cmd

        drive_px = min(dist_px - BALL_THRESHOLD_PX, MAX_DRIVE_PX)
        rotations = drive_px / calibration_pixels.ratio
        print(f"[FSM] Drive  {drive_px:.0f}px "
              f"(of {dist_px:.0f}px) → {rotations:.2f} rot  ({calibration_pixels.ratio:.2f} px/rot)")
        self._cal.record_drive(pose.px, rotations)
        robot.drive(rotations)
        self._pose.invalidate()
        return Command.FORWARD

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
            rotations = abs(err) / calibration_angle.ratio
            cmd = Command.RIGHT if err > 0 else Command.LEFT
            print(f"[FSM] Goal-align {cmd.name}  {abs(err):.1f}° → {rotations:.2f} rot  ({calibration_angle.ratio:.2f} deg/rot)")
            self._cal.record_turn(pose.angle, rotations)
            robot.turn(rotations, cmd.name)
            self._pose.invalidate()
            return cmd

        dist_px = _dist_px(pose.px, GOAL_POSITION_PX)
        if dist_px > GOAL_THRESHOLD_PX:
            drive_px = min(dist_px - GOAL_THRESHOLD_PX, MAX_DRIVE_PX)
            rotations = drive_px / calibration_pixels.ratio
            print(f"[FSM] Goal-drive  {drive_px:.0f}px (of {dist_px:.0f}px) → {rotations:.2f} rot  ({calibration_pixels.ratio:.2f} px/rot)")
            self._cal.record_drive(pose.px, rotations)
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
