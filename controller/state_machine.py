"""
state_machine.py — GolfBot complete pipeline controller.

Pipeline per ball:
    IDLE → ALIGN_BALL → DRIVE_BALL → COLLECT → ALIGN_GOAL → DRIVE_GOAL → RELEASE → (repeat)

World state format (from vision/tracker.py → extract_objects()):
    {
        "robot":       (x_cm, y_cm) or None,
        "white_balls": [(x_cm, y_cm), ...],
        "ob":          (x_cm, y_cm) or None,
        "robot_angle": float or None,           # from ArUco
    }
"""

import time
from enum import Enum, auto

import controller.ev3_controller as robot
from controller.navigation import angle_to_target, angle_error, nearest_ball, distance
from config import GOAL_POSITION_CM


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

ALIGN_THRESHOLD_DEG  = 10    # max heading error before driving
BALL_THRESHOLD_CM    = 12    # how close to ball before collecting
GOAL_THRESHOLD_CM    = 25    # how close to goal before releasing

POSE_TIMEOUT_S       = 0.5   # how long to trust last known position/angle
                              # if ArUco is lost for longer than this → stop


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE        = auto()   # pick next ball
    ALIGN_BALL  = auto()   # turn to face ball
    DRIVE_BALL  = auto()   # drive to ball
    COLLECT     = auto()   # pick up ball (claw)
    ALIGN_GOAL  = auto()   # turn to face goal
    DRIVE_GOAL  = auto()   # drive to goal
    RELEASE     = auto()   # release ball (gate)
    DONE        = auto()   # no balls left


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GolfBotController:

    def __init__(self):
        self.state         = State.IDLE
        self.target        = None              # current ball target
        self.goal          = GOAL_POSITION_CM  # hardcoded: far right, vertically centered
        self._last_command = None              # suppress duplicate TCP calls

        # Last known good pose — updated every time ArUco sees the marker
        self._last_pos      = None
        self._last_angle    = None
        self._last_seen_t   = 0.0              # timestamp of last valid ArUco detection

        print(f"[FSM] Ready. Goal: {self.goal}")

    # --- Public ---------------------------------------------------------------

    def update(self, world: dict) -> str:
        """
        Call once per frame with the latest world state from the vision team.
        Returns the command that was sent (used for debug overlay in main.py).
        """
        now = time.time()

        # Update pose cache whenever ArUco gives a valid reading
        raw_pos   = world.get("robot")
        raw_angle = world.get("robot_angle")

        if raw_pos is not None and raw_angle is not None:
            self._last_pos    = raw_pos
            self._last_angle  = raw_angle
            self._last_seen_t = now

        # Use cached pose — but only within the timeout window
        pose_age   = now - self._last_seen_t
        robot_pos  = self._last_pos   if pose_age < POSE_TIMEOUT_S else None
        robot_angle = self._last_angle if pose_age < POSE_TIMEOUT_S else None

        balls = self._all_balls(world)

        if robot_pos is None:
            # Marker lost too long — stop and wait
            return self._send("STOP")

        if self.state == State.IDLE:
            return self._idle(robot_pos, balls)

        if self.state == State.ALIGN_BALL:
            return self._align_to(robot_pos, robot_angle, self.target, next_state=State.DRIVE_BALL)

        if self.state == State.DRIVE_BALL:
            return self._drive_to(robot_pos, robot_angle, self.target,
                                  threshold=BALL_THRESHOLD_CM,
                                  next_state=State.COLLECT,
                                  align_back=State.ALIGN_BALL)

        if self.state == State.COLLECT:
            return self._collect()

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

    def _idle(self, robot_pos, balls) -> str:
        self.target = nearest_ball(robot_pos, balls)
        if self.target is None:
            print("[FSM] No balls left — done.")
            self._go(State.DONE)
        else:
            print(f"[FSM] New target: {self.target}")
            self._go(State.ALIGN_BALL)
        return self._send("STOP")

    def _align_to(self, robot_pos, robot_angle, target, next_state: State) -> str:
        if target is None or robot_angle is None:
            return self._send("STOP")

        error = angle_error(robot_angle, angle_to_target(robot_pos, target))
        print(f"[FSM] Aligning → {next_state.name} — error: {error:.1f}°")

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
        print(f"[FSM] Driving — {dist:.1f} cm  heading error: {error:.1f}°")

        if dist <= threshold:
            self._go(next_state)
            return self._send("STOP")

        if abs(error) > ALIGN_THRESHOLD_DEG * 2:
            self._go(align_back)
            return self._send("STOP")

        return self._send("FORWARD")

    def _collect(self) -> str:
        print("[FSM] Collecting ball...")
        robot.collect()
        print("[FSM] Ball collected.")
        self._go(State.ALIGN_GOAL)
        return "COLLECT"

    def _release(self) -> str:
        print("[FSM] Releasing ball...")
        robot.release()
        print("[FSM] Ball released.")
        self.target = None
        self._go(State.IDLE)
        return "RELEASE"

    # --- Helpers --------------------------------------------------------------

    def _send(self, command: str) -> str:
        """Send command to robot only if it changed since last frame."""
        if command != self._last_command:
            if command == "FORWARD":  robot.drive()
            elif command == "LEFT":   robot.turn_left()
            elif command == "RIGHT":  robot.turn_right()
            elif command == "STOP":   robot.stop()
            self._last_command = command
        return command

    def _go(self, new_state: State):
        print(f"[FSM] {self.state.name} → {new_state.name}")
        self.state = new_state

    @staticmethod
    def _all_balls(world: dict) -> list:
        balls = list(world.get("white_balls", []))
        if world.get("ob"):
            balls.append(world["ob"])
        return balls
