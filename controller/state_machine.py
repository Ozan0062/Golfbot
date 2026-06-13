"""
state_machine.py — GolfBot decision logic.

STATE FLOW
──────────
  SEEK  →  lock onto next ball from TSP route
    ↓
  ALIGN  →  turn to face the locked target
    ↓ heading within 10°
  APPROACH  →  drive toward target (capped per step)
    ↓ after each drive step, go back to ALIGN
    ↓ when within collect radius
  (collect ball, clear target, back to SEEK)

  When SEEK finds no balls:
  REVERSE_WHITE  →  back up, re-scan for white or orange
    ↓ still nothing
  REVERSE_ORANGE  →  back up, re-scan for orange only
    ↓ still nothing (all collected)
  DRIVE_GOAL  →  turn + drive to goal zone
    ↓ arrived
  RELEASE  →  open gate
    ↓
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
from controller.navigation import angle_to_target, angle_error
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager, RouteTarget
from config import GOAL_POSITION_CM, GOAL_POSITION_PX


# ─── Thresholds ──────────────────────────────────────────────────────────────

ALIGN_THRESHOLD_DEG = 10    # Below this, we consider ourselves "aligned" and drive straight.
MIN_TURN_ROTATIONS = 0.25   # Ignore turns smaller than this.
TURN_DAMPING = 0.6          # Reduce turn commands to prevent oscillation when close.

COLLECT_RADIUS_PX = 15      # Accepted radius for collecting a ball.
GOAL_THRESHOLD_PX = 30      # Close enough to goal to stop driving and release balls.
REVERSE_ROTATIONS = 1.5     # How far to reverse when no balls are visible.
MAX_DRIVE_PX = 80           # Cap on drive distance per cycle to allow for course correction.


# ─── States ──────────────────────────────────────────────────────────────────

class State(Enum):
    SEEK           = auto()   # acquire next target from TSP route
    ALIGN          = auto()   # turn to face the locked target
    APPROACH       = auto()   # drive toward the locked target
    REVERSE_WHITE  = auto()   # back up, scan for white or orange balls
    REVERSE_ORANGE = auto()   # back up, scan for orange ball only
    DRIVE_GOAL     = auto()   # navigate to goal zone (turn + drive)
    RELEASE        = auto()   # dump balls at goal
    DONE           = auto()   # mission complete


# ─── Controller ──────────────────────────────────────────────────────────────

class GolfBotController:

    def __init__(self):
        self.state           = State.SEEK
        self._pose           = PoseCache()
        self._route          = RouteManager()
        self._cal            = CalibrationManager()
        self._has_reversed   = False
        self._locked_target  = None   # type: RouteTarget | None
        print(f"[FSM] Ready.  Goal={GOAL_POSITION_CM}  "
              f"cal={calibration_angle.ratio:.1f}deg/rot  {calibration_pixels.ratio:.1f}px/rot")

    # ── Main entry point (called once per camera frame) ──────────────────────

    def update(self, world: dict) -> Command:
        """Run one tick of the state machine. Returns the command executed."""
        pose = self._pose.update(world)
        if pose is None:
            print("[FSM] ArUco not detected — waiting.")
            return Command.STOP

        self._cal.consume(pose.px, pose.angle)

        # ── State dispatch ───────────────────────────────────────────────
        if self.state == State.SEEK:
            return self._seek(pose, world)
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

    # ─── State: SEEK ─────────────────────────────────────────────────────────
    # Lock onto the next ball. Immediate transition — no motor commands here.

    def _seek(self, pose, world) -> Command:
        self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        if self._locked_target is None:
            print("[FSM] No balls visible — reversing.")
            self._route.clear()
            self._has_reversed = False
            self._transition(State.REVERSE_WHITE)
            return Command.STOP

        target = self._locked_target
        print(f"[SEEK] Locked target at ({target.px[0]:.0f},{target.px[1]:.0f})px")
        self._transition(State.ALIGN)
        return Command.STOP

    # ─── State: ALIGN ────────────────────────────────────────────────────────
    # Turn to face the locked target. Once heading error is within threshold,
    # transition to APPROACH. Each frame either turns or passes through.

    def _align(self, pose) -> Command:
        target = self._locked_target
        if target is None:
            # Target lost (shouldn't happen, but recover gracefully)
            self._transition(State.SEEK)
            return Command.STOP

        heading_error = self._heading_error_to(pose, target.cm)
        print(f"[ALIGN] heading={pose.angle:.1f}°  error={heading_error:.1f}°")

        if abs(heading_error) <= ALIGN_THRESHOLD_DEG:
            # Aligned — move on to driving
            self._transition(State.APPROACH)
            return Command.STOP

        # Turn toward target
        rotations = _angle_to_rotations(heading_error)
        if rotations < MIN_TURN_ROTATIONS:
            # Turn too small to execute — close enough, just drive
            self._transition(State.APPROACH)
            return Command.STOP

        direction = Command.RIGHT if heading_error > 0 else Command.LEFT
        print(f"[ALIGN] Turn {direction.name}  {abs(heading_error):.1f}° → {rotations:.2f}rot")
        self._execute_turn(pose, rotations, direction)
        return direction
        # Stay in ALIGN — next frame will re-check heading

    # ─── State: APPROACH ─────────────────────────────────────────────────────
    # Drive toward the locked target. After each drive step, go back to ALIGN
    # to re-check heading. When close enough, collect and return to SEEK.

    def _approach(self, pose) -> Command:
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        dist_px = _distance_px(pose.px, target.px)
        print(f"[APPROACH] dist={dist_px:.0f}px  target=({target.px[0]:.0f},{target.px[1]:.0f})px")

        # Close enough → collect
        if dist_px <= COLLECT_RADIUS_PX:
            print(f"[APPROACH] Within collect radius — grabbing ball")
            self._locked_target = None
            self._route.advance()
            robot.collect()
            self._pose.invalidate()
            self._transition(State.SEEK)
            return Command.COLLECT

        # Drive forward (capped), then re-align
        drive_px  = min(dist_px - COLLECT_RADIUS_PX, MAX_DRIVE_PX)
        rotations = _px_to_rotations(drive_px)
        print(f"[APPROACH] Drive {drive_px:.0f}px → {rotations:.2f}rot")
        self._execute_drive(pose, rotations)
        self._transition(State.ALIGN)
        return Command.FORWARD
        # Next frame enters ALIGN to re-check heading before driving again

    # ─── State: REVERSE (shared by REVERSE_WHITE and REVERSE_ORANGE) ─────────

    def _handle_reverse(self, world, scan_for: tuple, next_if_found: State,
                        next_if_empty: State) -> Command:
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

    # ─── State: DRIVE_GOAL ───────────────────────────────────────────────────
    # Same turn-then-drive pattern, but toward the fixed goal position.

    def _drive_to_goal(self, pose) -> Command:
        heading_error = self._heading_error_to(pose, GOAL_POSITION_CM)

        # Turn to face goal if needed
        if abs(heading_error) > ALIGN_THRESHOLD_DEG:
            rotations = _angle_to_rotations(heading_error)
            if rotations >= MIN_TURN_ROTATIONS:
                direction = Command.RIGHT if heading_error > 0 else Command.LEFT
                print(f"[GOAL] Turn {direction.name}  {abs(heading_error):.1f}° → {rotations:.2f}rot")
                self._execute_turn(pose, rotations, direction)
                return direction

        # Drive toward goal
        dist_px = _distance_px(pose.px, GOAL_POSITION_PX)
        if dist_px > GOAL_THRESHOLD_PX:
            drive_px  = min(dist_px - GOAL_THRESHOLD_PX, MAX_DRIVE_PX)
            rotations = _px_to_rotations(drive_px)
            print(f"[GOAL] Drive {drive_px:.0f}px → {rotations:.2f}rot")
            self._execute_drive(pose, rotations)
            return Command.FORWARD

        # Arrived
        print("[GOAL] At goal — releasing.")
        self._transition(State.RELEASE)
        return Command.STOP

    # ─── State: RELEASE ──────────────────────────────────────────────────────

    def _release_balls(self) -> Command:
        robot.release()
        self._transition(State.DONE)
        return Command.RELEASE

    # ─── Movement helpers ────────────────────────────────────────────────────

    def _execute_drive(self, pose, rotations: float):
        self._cal.record_drive(pose.px, rotations)
        robot.drive(rotations)
        self._pose.invalidate()

    def _execute_turn(self, pose, rotations: float, direction: Command):
        self._cal.record_turn(pose.angle, rotations)
        robot.turn(rotations, direction.name)
        self._pose.invalidate()

    def _heading_error_to(self, pose, target_cm: tuple) -> float:
        bearing = angle_to_target(pose.pos, target_cm)
        return angle_error(pose.angle, bearing)

    def _transition(self, new_state: State):
        print(f"[FSM] {self.state.name} → {new_state.name}")
        self.state = new_state


# ─── Module-level helpers ────────────────────────────────────────────────────

def _distance_px(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _px_to_rotations(drive_px: float) -> float:
    return drive_px / calibration_pixels.ratio


def _angle_to_rotations(heading_error: float) -> float:
    return abs(heading_error) / calibration_angle.ratio * TURN_DAMPING
