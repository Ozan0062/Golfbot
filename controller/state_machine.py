"""
state_machine.py — GolfBot MVP controller.

Pipeline:
    FIND_BALL → ALIGN_BALL → DRIVE_BALL → COLLECT → (repeat until no balls)
    → REVERSE 1s → FIND_BALL → (still none?) → ALIGN_GOAL → DRIVE_GOAL → RELEASE → DONE
"""

import time
from enum import Enum, auto

import controller.ev3_controller as robot
from controller.navigation import angle_to_target, angle_error, nearest_ball, distance
from config import GOAL_POSITION_CM


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

ALIGN_THRESHOLD_DEG = 10    # max heading error before driving
BALL_THRESHOLD_CM   = 10
# how close before collecting
GOAL_THRESHOLD_CM   = 25    # how close to goal before releasing
POSE_TIMEOUT_S      = 0.5   # stop if ArUco lost longer than this
REVERSE_DURATION_S  = 1.0   # how long to reverse when no balls found


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    FIND_BALL  = auto()   # pick nearest ball
    ALIGN_BALL = auto()   # turn to face it
    DRIVE_BALL = auto()   # drive to it
    COLLECT    = auto()   # close claw
    REVERSE    = auto()   # back up and re-check
    ALIGN_GOAL = auto()   # turn to face goal
    DRIVE_GOAL = auto()   # drive to goal
    RELEASE    = auto()   # open gate
    DONE       = auto()   # finished


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GolfBotController:

    def __init__(self):
        self.state          = State.FIND_BALL
        self.target         = None
        self.goal           = GOAL_POSITION_CM
        self._last_command  = None
        self._last_pos      = None
        self._last_angle    = None
        self._last_seen_t   = 0.0
        self._reverse_start = None
        print(f"[FSM] Ready. Goal: {self.goal}")

    def update(self, world: dict) -> str:
        """Call once per frame. Returns the command sent (for debug overlay)."""
        now = time.time()

        # Update pose cache whenever ArUco gives a valid reading
        if world.get("robot") is not None and world.get("robot_angle") is not None:
            self._last_pos    = world["robot"]
            self._last_angle  = world["robot_angle"]
            self._last_seen_t = now

        pose_age    = now - self._last_seen_t
        robot_pos   = self._last_pos   if pose_age < POSE_TIMEOUT_S else None
        robot_angle = self._last_angle if pose_age < POSE_TIMEOUT_S else None
        balls       = _all_balls(world)

        if robot_pos is None:
            return self._send("STOP")

        if self.state == State.FIND_BALL:
            return self._find_ball(robot_pos, balls)

        if self.state == State.ALIGN_BALL:
            return self._align_to(robot_pos, robot_angle, self.target, next_state=State.DRIVE_BALL)

        if self.state == State.DRIVE_BALL:
            return self._drive_to(robot_pos, robot_angle, self.target,
                                  threshold=BALL_THRESHOLD_CM,
                                  next_state=State.COLLECT,
                                  align_back=State.ALIGN_BALL)

        if self.state == State.COLLECT:
            return self._collect()

        if self.state == State.REVERSE:
            return self._reverse(now, balls)

        if self.state == State.ALIGN_GOAL:
            return self._align_to(robot_pos, robot_angle, self.goal, next_state=State.DRIVE_GOAL)

        if self.state == State.DRIVE_GOAL:
            return self._drive_to(robot_pos, robot_angle, self.goal,
                                  threshold=GOAL_THRESHOLD_CM,
                                  next_state=State.RELEASE,
                                  align_back=State.ALIGN_GOAL)

        if self.state == State.RELEASE:
            return self._release()

        if self.state == State.DONE:
            return self._send("STOP")

        return self._send("STOP")

    # --- State handlers -------------------------------------------------------

    def _find_ball(self, robot_pos, balls) -> str:
        self.target = nearest_ball(robot_pos, balls)
        if self.target is None:
            print("[FSM] No balls visible — reversing.")
            self._reverse_start = None
            self._go(State.REVERSE)
        else:
            print(f"[FSM] Target: {self.target}")
            self._go(State.ALIGN_BALL)
        return self._send("STOP")

    def _reverse(self, now: float, balls: list) -> str:
        if self._reverse_start is None:
            self._reverse_start = now

        if now - self._reverse_start < REVERSE_DURATION_S:
            return self._send("BACKWARD")

        # Done reversing — check for balls again
        self._reverse_start = None
        if balls:
            print("[FSM] Balls found after reverse — resuming.")
            self._go(State.FIND_BALL)
        else:
            print("[FSM] No balls after reverse — heading to goal.")
            self._go(State.ALIGN_GOAL)
        return self._send("STOP")

    def _align_to(self, robot_pos, robot_angle, target, next_state: State) -> str:
        if target is None or robot_angle is None:
            return self._send("STOP")
        error = angle_error(robot_angle, angle_to_target(robot_pos, target))
        print(f"[FSM] Aligning → {next_state.name}  error: {error:.1f}°")
        if abs(error) <= ALIGN_THRESHOLD_DEG:
            self._go(next_state)
            return self._send("STOP")
        return self._send("RIGHT" if error > 0 else "LEFT")

    def _drive_to(self, robot_pos, robot_angle, target, threshold: float,
                  next_state: State, align_back: State) -> str:
        if target is None or robot_angle is None:
            return self._send("STOP")
        dist  = distance(robot_pos, target)
        error = angle_error(robot_angle, angle_to_target(robot_pos, target))
        print(f"[FSM] Driving — {dist:.1f} cm  error: {error:.1f}°")
        if dist <= threshold:
            self._go(next_state)
            return self._send("STOP")
        if abs(error) > ALIGN_THRESHOLD_DEG * 2:
            self._go(align_back)
            return self._send("STOP")
        return self._send("FORWARD")

    def _collect(self) -> str:
        print("[FSM] Collecting...")
        robot.collect()
        print("[FSM] Done. Finding next ball.")
        self._go(State.FIND_BALL)
        return "COLLECT"

    def _release(self) -> str:
        print("[FSM] Releasing...")
        robot.release()
        print("[FSM] Done.")
        self._go(State.DONE)
        return "RELEASE"

    # --- Helpers --------------------------------------------------------------

    def _send(self, command: str) -> str:
        if command != self._last_command:
            if   command == "FORWARD":  robot.drive()
            elif command == "BACKWARD": robot.reverse()
            elif command == "LEFT":     robot.turn_left()
            elif command == "RIGHT":    robot.turn_right()
            elif command == "STOP":     robot.stop()
            self._last_command = command
        return command

    def _go(self, new_state: State):
        print(f"[FSM] {self.state.name} → {new_state.name}")
        self.state = new_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_balls(world: dict) -> list:
    """Combine white balls and orange ball into one list — no colour preference in MVP."""
    balls = list(world.get("white_balls", []))
    if world.get("ob"):
        balls.append(world["ob"])
    return balls
