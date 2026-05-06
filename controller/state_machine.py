"""
state_machine.py — GolfBot MVP controller.

Receives world state from the vision team every frame and decides what
command to send to the robot.

World state format (from vision/tracker.py → extract_objects()):
    {
        "robot":       (x_cm, y_cm) or None,
        "white_balls": [(x_cm, y_cm), ...],
        "ob":          (x_cm, y_cm) or None,   # orange ball
        "cross":       (x_cm, y_cm) or None,   # cross/obstacle marker
    }
"""

from enum import Enum, auto

import controller.ev3_controller as robot
from controller.navigation import angle_to_target, angle_error, nearest_ball, distance


# ---------------------------------------------------------------------------
# Tuning constants — adjust on the real field
# ---------------------------------------------------------------------------

ALIGN_THRESHOLD_DEG  = 10   # max heading error before we start driving
ARRIVAL_THRESHOLD_CM = 15   # distance at which a ball counts as collected


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE  = auto()   # pick the next target
    ALIGN = auto()   # turn to face target
    DRIVE = auto()   # drive toward target
    DONE  = auto()   # no balls left


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GolfBotController:

    def __init__(self):
        self.state  = State.IDLE
        self.target = None
        robot.reset_angle()
        print("[FSM] Ready.")

    # --- Public ---------------------------------------------------------------

    def update(self, world: dict) -> str:
        """
        Call once per frame with the latest world state from the vision team.
        Returns the command that was sent (useful for debug overlay in main.py).
        """
        robot_pos = world.get("robot")
        balls     = self._all_balls(world)

        if robot_pos is None:
            robot.stop()
            return "STOP"

        if self.state == State.IDLE:
            return self._idle(robot_pos, balls)

        if self.state == State.ALIGN:
            return self._align(robot_pos)

        if self.state == State.DRIVE:
            return self._drive(robot_pos)

        if self.state == State.DONE:
            robot.stop()
            return "STOP"

        return "STOP"

    # --- State handlers -------------------------------------------------------

    def _idle(self, robot_pos, balls) -> str:
        """Pick the nearest ball and transition to ALIGN."""
        self.target = nearest_ball(robot_pos, balls)
        if self.target is None:
            print("[FSM] No balls found — done.")
            self._go(State.DONE)
        else:
            print(f"[FSM] Target: {self.target}")
            self._go(State.ALIGN)
        robot.stop()
        return "STOP"

    def _align(self, robot_pos) -> str:
        """Turn in place until we face the target."""
        current = robot.get_angle()
        desired = angle_to_target(robot_pos, self.target)
        error   = angle_error(current, desired)

        print(f"[FSM] Aligning — error: {error:.1f}°")

        if abs(error) <= ALIGN_THRESHOLD_DEG:
            self._go(State.DRIVE)
            robot.stop()
            return "STOP"

        if error > 0:
            robot.turn_right()
            return "RIGHT"
        else:
            robot.turn_left()
            return "LEFT"

    def _drive(self, robot_pos) -> str:
        """Drive straight toward the target, re-align if heading drifts."""
        dist    = distance(robot_pos, self.target)
        current = robot.get_angle()
        desired = angle_to_target(robot_pos, self.target)
        error   = angle_error(current, desired)

        print(f"[FSM] Driving — {dist:.1f} cm  heading error: {error:.1f}°")

        if dist <= ARRIVAL_THRESHOLD_CM:
            print("[FSM] Ball collected.")
            self.target = None
            self._go(State.IDLE)
            robot.stop()
            return "STOP"

        if abs(error) > ALIGN_THRESHOLD_DEG * 2:
            self._go(State.ALIGN)
            robot.stop()
            return "STOP"

        robot.drive()
        return "FORWARD"

    # --- Helpers --------------------------------------------------------------

    def _go(self, new_state: State):
        print(f"[FSM] {self.state.name} → {new_state.name}")
        self.state = new_state

    @staticmethod
    def _all_balls(world: dict) -> list:
        """Combine white balls and orange ball into one list."""
        balls = list(world.get("white_balls", []))
        if world.get("ob"):
            balls.append(world["ob"])
        return balls
