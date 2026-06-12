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

ALIGN_THRESHOLD_DEG = 15    # degrees — turn if heading error exceeds this
MIN_TURN_ROTATIONS  = 0.25  # skip turns smaller than this — prevents oscillation
COLLECT_RADIUS_PX   = 10    # pixels  — collect if closer than this
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
        print(f"[FSM] Ready.  Goal={GOAL_POSITION_CM}  "
              f"cal={calibration_angle.ratio:.1f}deg/rot  {calibration_pixels.ratio:.1f}px/rot")

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
        # ── 1. Lock target ──────────────────────────────────────────────
        if self._locked_target is None:
            self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        if self._locked_target is None:
            print("[FSM] No balls visible — reversing.")
            self._route.clear()
            self._reversed = False
            self._go(State.REVERSE_WHITE)
            return Command.STOP

        target = self._locked_target

        # ── 2. Measure error every cycle (handles drift) ────────────────
        dist_px        = _dist_px(pose.px, target.px)
        target_bearing = angle_to_target(pose.pos, target.cm)
        err            = angle_error(pose.angle, target_bearing)

        print(f"[SEEK] robot=({pose.px[0]:.0f},{pose.px[1]:.0f})px heading={pose.angle:.1f}°  "
              f"target=({target.px[0]:.0f},{target.px[1]:.0f})px  "
              f"err={err:.1f}°  dist={dist_px:.0f}px")

        # ── 3. Close enough → collect ───────────────────────────────────
        if dist_px <= COLLECT_RADIUS_PX:
            print(f"[FSM] Collect — dist={dist_px:.0f}px")
            self._locked_target = None
            self._route.advance()
            robot.collect()
            self._pose.invalidate()
            return Command.COLLECT

        # ── 4. Rotate until aligned ─────────────────────────────────────
        if abs(err) > ALIGN_THRESHOLD_DEG:
            rotations = abs(err) / calibration_angle.ratio
            if rotations < MIN_TURN_ROTATIONS:
                # Deadband — skip tiny turns that cause oscillation
                pass
            else:
                cmd = Command.RIGHT if err > 0 else Command.LEFT
                print(f"[FSM] Turn {cmd.name}  {abs(err):.1f}° → {rotations:.2f}rot")
                self._cal.record_turn(pose.angle, rotations)
                robot.turn(rotations, cmd.name)
                self._pose.invalidate()
                return cmd

        # ── 5. Drive forward (capped, re-check next cycle) ─────────────
        drive_px  = min(dist_px - COLLECT_RADIUS_PX, MAX_DRIVE_PX)
        rotations = drive_px / calibration_pixels.ratio
        print(f"[FSM] Drive  {drive_px:.0f}px → {rotations:.2f}rot")
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
            if rotations >= MIN_TURN_ROTATIONS:
                cmd = Command.RIGHT if err > 0 else Command.LEFT
                print(f"[FSM] Goal-align {cmd.name}  {abs(err):.1f}° → {rotations:.2f}rot")
                self._cal.record_turn(pose.angle, rotations)
                robot.turn(rotations, cmd.name)
                self._pose.invalidate()
                return cmd

        dist_px = _dist_px(pose.px, GOAL_POSITION_PX)
        if dist_px > GOAL_THRESHOLD_PX:
            drive_px = min(dist_px - GOAL_THRESHOLD_PX, MAX_DRIVE_PX)
            rotations = drive_px / calibration_pixels.ratio
            print(f"[FSM] Goal-drive  {drive_px:.0f}px → {rotations:.2f}rot")
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
