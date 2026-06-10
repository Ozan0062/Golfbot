"""
state_machine.py — GolfBot controller, per revised flowchart.

Each update() call is one camera frame. Blocking EV3 moves (drive/turn)
pause the loop while the brick executes; the next call sees updated positions.

Main loop (SEEK state) each frame:
  1. Find target: white balls first, then orange ball
  2. No target → REVERSE_WHITE
  3. Angle error > 10° → turn (blocking), record pending angle calibration
  4. Distance > BALL_THRESHOLD_CM → drive (blocking), record pending drive calibration
  5. Close enough → collect → back to SEEK

Calibration happens at the top of the next frame using the fresh ArUco heading/position.

Reverse logic:
  REVERSE_WHITE  → reverse once, next frame check white+orange → SEEK or REVERSE_ORANGE
  REVERSE_ORANGE → reverse once, next frame check orange → SEEK or DRIVE_GOAL
"""

import time
from enum import Enum, auto

import controller.ev3_controller as robot
from controller.navigation import angle_to_target, angle_error, nearest_ball, distance
from controller.calibration import (
    rotations_for_angle,
    rotations_for_distance,
    measure_degrees_per_rotation,
    measure_cm_per_rotation,
)
from controller.calibration_tracker import calibration_pixels, calibration_angle
from config import GOAL_POSITION_CM, GOAL_POSITION_PX


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

ALIGN_THRESHOLD_DEG  = 10    # turn if heading error exceeds this (degrees)
BALL_THRESHOLD_PX    = 10    # collect if within this distance (pixels)
GOAL_THRESHOLD_PX    = 60    # release at goal if within this distance (pixels)
REVERSE_ROTATIONS    = 1.5   # motor rotations per reverse manoeuvre
POSE_TIMEOUT_S       = 0.5   # use cached ArUco pose if marker not seen for this long


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    SEEK           = auto()   # find target, align, drive, or collect
    REVERSE_WHITE  = auto()   # reverse then re-check for white balls
    REVERSE_ORANGE = auto()   # reverse then re-check for orange ball
    DRIVE_GOAL     = auto()   # align to goal and drive to it
    RELEASE        = auto()   # open gate
    DONE           = auto()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GolfBotController:

    def __init__(self):
        self.state          = State.SEEK
        self.target         = None
        self._reversed      = False   # flag: have we done the reverse this state?
        self._pending_drive = None    # (start_px, rotations) — calibrate next frame
        self._pending_turn  = None    # (start_angle_deg, rotations) — calibrate next frame
        # Pose cache — ArUco can miss frames; keep last known pose for POSE_TIMEOUT_S
        self._last_pos      = None
        self._last_px       = None
        self._last_angle    = None
        self._last_seen_t   = 0.0
        print(f"[FSM] Ready.  Goal: {GOAL_POSITION_CM}")

    def update(self, world: dict) -> str:
        """Called once per camera frame. Returns last command string for overlay."""
        now = time.time()

        # Update pose cache whenever ArUco gives a valid reading
        if world.get("robot") is not None and world.get("robot_angle") is not None:
            self._last_pos    = world["robot"]
            self._last_px     = world.get("robot_px")
            self._last_angle  = world["robot_angle"]
            self._last_seen_t = now

        pose_age    = now - self._last_seen_t
        robot_pos   = self._last_pos   if pose_age < POSE_TIMEOUT_S else None
        robot_px    = self._last_px    if pose_age < POSE_TIMEOUT_S else None
        robot_angle = self._last_angle if pose_age < POSE_TIMEOUT_S else None

        if robot_pos is None:
            print("[FSM] ArUco not detected — waiting.")
            return "STOP"

        # Consume calibration from previous move (fresh ArUco data now available)
        self._calibrate_drive(robot_px)
        self._calibrate_turn(robot_angle)

        if self.state == State.SEEK:
            return self._seek(robot_pos, robot_px, robot_angle, world)

        if self.state == State.REVERSE_WHITE:
            return self._reverse_white(world)

        if self.state == State.REVERSE_ORANGE:
            return self._reverse_orange(world)

        if self.state == State.DRIVE_GOAL:
            return self._drive_goal(robot_pos, robot_px, robot_angle)

        if self.state == State.RELEASE:
            robot.release()
            self._go(State.DONE)
            return "RELEASE"

        # DONE
        return "STOP"

    # -------------------------------------------------------------------------
    # State handlers
    # -------------------------------------------------------------------------

    def _seek(self, robot_pos, robot_px, robot_angle, world) -> str:
        """Find nearest target, align, drive, or collect."""

        white_balls    = world.get("white_balls", [])
        white_balls_px = world.get("white_balls_px", [])
        orange_ball    = world.get("ob")
        orange_ball_px = world.get("ob_px")

        # Priority: white balls first, then orange
        # Use cm positions for angle maths; px positions for distance/drive
        target    = nearest_ball(robot_pos, white_balls) if white_balls else None
        target_px = white_balls_px[white_balls.index(target)] if target is not None else None
        if target is None and orange_ball:
            target    = orange_ball
            target_px = orange_ball_px

        if target is None:
            print("[FSM] No balls visible — reversing.")
            self._reversed = False
            self._go(State.REVERSE_WHITE)
            return "STOP"

        self.target = target

        # ── Align? (angle uses cm coords — scale-invariant) ─────────────────
        desired = angle_to_target(robot_pos, target)
        err     = angle_error(robot_angle, desired)

        if abs(err) > ALIGN_THRESHOLD_DEG:
            rotations = rotations_for_angle(abs(err), calibration_angle.ratio)
            direction  = "RIGHT" if err > 0 else "LEFT"
            print(f"[FSM] Turn {direction}  err={err:.1f}°  rot={rotations:.2f}")

            self._pending_turn = (robot_angle, rotations)
            robot.turn(rotations, direction)          # blocking; calibrate next frame
            return "TURN"

        # ── Drive? (distance and calibration in pixels) ──────────────────────
        dist_px = distance(robot_px, target_px) if robot_px and target_px else 0
        if dist_px > BALL_THRESHOLD_PX:
            rotations = rotations_for_distance(dist_px, calibration_pixels.ratio)
            print(f"[FSM] Drive  dist={dist_px:.0f}px  rot={rotations:.2f}")

            self._pending_drive = (robot_px, rotations)
            robot.drive(rotations)                    # blocking; calibrate next frame
            return "FORWARD"

        # ── Collect ─────────────────────────────────────────────────────────
        print("[FSM] Collecting.")
        robot.collect()
        return "COLLECT"

    def _reverse_white(self, world) -> str:
        if not self._reversed:
            print("[FSM] Reversing (looking for white balls).")
            robot.reverse(REVERSE_ROTATIONS)          # blocking
            self._reversed = True
            return "BACKWARD"

        # Next frame: fresh ball detections
        self._reversed = False
        if world.get("white_balls") or world.get("ob"):
            print("[FSM] Balls found after reverse.")
            self._go(State.SEEK)
        else:
            print("[FSM] Still no white — reversing for orange.")
            self._go(State.REVERSE_ORANGE)
        return "STOP"

    def _reverse_orange(self, world) -> str:
        if not self._reversed:
            print("[FSM] Reversing (looking for orange ball).")
            robot.reverse(REVERSE_ROTATIONS)          # blocking
            self._reversed = True
            return "BACKWARD"

        self._reversed = False
        if world.get("ob"):
            print("[FSM] Orange ball found after reverse.")
            self._go(State.SEEK)
        else:
            print("[FSM] No balls — heading to goal.")
            self._go(State.DRIVE_GOAL)
        return "STOP"

    def _drive_goal(self, robot_pos, robot_px, robot_angle) -> str:
        """Align to goal then drive to it; release when close enough."""

        # Angle uses cm (scale-invariant); distance/drive uses pixels
        desired = angle_to_target(robot_pos, GOAL_POSITION_CM)
        err     = angle_error(robot_angle, desired)

        if abs(err) > ALIGN_THRESHOLD_DEG:
            rotations = rotations_for_angle(abs(err), calibration_angle.ratio)
            direction  = "RIGHT" if err > 0 else "LEFT"
            print(f"[FSM] Goal align {direction}  err={err:.1f}°")

            self._pending_turn = (robot_angle, rotations)
            robot.turn(rotations, direction)          # blocking; calibrate next frame
            return "TURN"

        dist_px = distance(robot_px, GOAL_POSITION_PX) if robot_px else 0
        if dist_px > GOAL_THRESHOLD_PX:
            rotations = rotations_for_distance(dist_px, calibration_pixels.ratio)
            print(f"[FSM] Drive to goal  dist={dist_px:.0f}px  rot={rotations:.2f}")

            self._pending_drive = (robot_px, rotations)
            robot.drive(rotations)                    # blocking; calibrate next frame
            return "FORWARD"

        print("[FSM] At goal — releasing.")
        self._go(State.RELEASE)
        return "STOP"

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calibrate_drive(self, robot_px):
        """Compare ArUco pixel position before/after a drive move to refine px/rotation."""
        if self._pending_drive is None or robot_px is None:
            return
        start_px, rotations = self._pending_drive
        self._pending_drive  = None
        if rotations > 0:
            measured = measure_cm_per_rotation(start_px, robot_px, rotations)  # reuses dist helper
            calibration_pixels.update(measured)
            print(f"[CAL] px/rot → {calibration_pixels.ratio:.2f}")

    def _calibrate_turn(self, robot_angle):
        """Compare ArUco heading before/after a turn to refine degrees/rotation."""
        if self._pending_turn is None or robot_angle is None:
            return
        start_angle, rotations = self._pending_turn
        self._pending_turn     = None
        if rotations > 0:
            measured = measure_degrees_per_rotation(start_angle, robot_angle, rotations)
            calibration_angle.update(measured)
            print(f"[CAL] deg/rot → {calibration_angle.ratio:.2f}")

    def _go(self, new_state: State):
        print(f"[FSM] {self.state.name} → {new_state.name}")
        self.state = new_state
