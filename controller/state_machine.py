"""
state_machine.py — GolfBot decision logic.

This file contains only state transitions and decisions.
All implementation details live in dedicated modules:

    pose_cache.py          ArUco timeout / pose freshness
    route_manager.py       TSP route, ball gathering, px lookup
    calibration_manager.py deferred calibration after blocking moves
    calibration_tracker.py EMA ratio trackers
    calibration.py         rotation math (angle → rotations, distance → rotations)
    tsp_christofides.py    Christofides algorithm
    navigation.py          angle_to_target, angle_error
    ev3_controller.py      robot hardware commands
    commands.py            Command enum
"""

from enum import Enum, auto

import controller.ev3_controller as robot
from controller.calibration import rotations_for_angle, rotations_for_distance
from controller.calibration_tracker import calibration_pixels, calibration_angle
from controller.calibration_manager import CalibrationManager
from controller.commands import Command
from controller.navigation import angle_to_target, angle_error
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager
from config import GOAL_POSITION_CM, GOAL_POSITION_PX


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ALIGN_THRESHOLD_DEG = 10    # degrees — turn if heading error exceeds this
BALL_THRESHOLD_PX   = 25    # pixels  — collect if closer than this
GOAL_THRESHOLD_PX   = 30    # pixels  — release when this close to goal
REVERSE_ROTATIONS   = 1.5   # motor rotations per reverse manoeuvre


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
        self.state    = State.SEEK
        self._pose    = PoseCache()
        self._route   = RouteManager()
        self._cal     = CalibrationManager()
        self._reversed = False
        print(f"[FSM] Ready.  Goal: {GOAL_POSITION_CM}")

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
        target = self._route.get_target(pose.pos, pose.px, world) 

        if target is None:
            print("[FSM] No balls visible — reversing.")
            self._route.clear()
            self._reversed = False
            self._go(State.REVERSE_WHITE)
            return Command.STOP

        err = angle_error(pose.angle, angle_to_target(pose.pos, target.cm))

        if abs(err) > ALIGN_THRESHOLD_DEG:
            rotations = rotations_for_angle(abs(err), calibration_angle.ratio)
            cmd = Command.RIGHT if err > 0 else Command.LEFT
            self._cal.record_turn(pose.angle, rotations)
            robot.turn(rotations, cmd.name)
            return cmd

        if target.dist_px > BALL_THRESHOLD_PX:
            rotations = rotations_for_distance(target.dist_px, calibration_pixels.ratio)
            self._cal.record_drive(pose.px, rotations)
            robot.drive(rotations)
            return Command.FORWARD

        self._route.advance()
        robot.collect()
        return Command.COLLECT

    def _reverse_white(self, world) -> Command:
        if not self._reversed:
            robot.reverse(REVERSE_ROTATIONS)
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
            rotations = rotations_for_angle(abs(err), calibration_angle.ratio)
            cmd = Command.RIGHT if err > 0 else Command.LEFT
            self._cal.record_turn(pose.angle, rotations)
            robot.turn(rotations, cmd.name)
            return cmd

        dist_px = _dist_px(pose.px, GOAL_POSITION_PX)
        if dist_px > GOAL_THRESHOLD_PX:
            rotations = rotations_for_distance(dist_px, calibration_pixels.ratio)
            self._cal.record_drive(pose.px, rotations)
            robot.drive(rotations)
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
