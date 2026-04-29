"""
state_machine.py — GolfBot MVP controller.
Finds the nearest ball, aligns, drives to it, repeats until none left.
"""

from enum import Enum, auto
from controller.navigation import angle_to_target, angle_error, nearest_ball, distance
from ev3 import ev3_controller as robot

ALIGN_THRESHOLD_DEG  = 10   # degrees — close enough to stop turning
ARRIVAL_THRESHOLD_CM = 15   # cm — close enough to count as collected


class State(Enum):
    IDLE  = auto()
    ALIGN = auto()
    DRIVE = auto()
    DONE  = auto()


class GolfBotController:
    def __init__(self):
        self.state  = State.IDLE
        self.target = None
        robot.reset_angle(0)
        print("[Controller] Ready.")

    def update(self, world_state):
        """Call once per frame. Returns the command sent to the robot."""
        if self.state == State.DONE:
            return "STOP"

        robot_pos  = world_state.get("robot")
        all_balls  = world_state.get("white_balls", [])

        if world_state.get("ob"):
            all_balls = all_balls + [world_state["ob"]]

        if robot_pos is None:
            robot.send_command("STOP")
            return "STOP"

        # --- IDLE: pick a target ---
        if self.state == State.IDLE:
            self.target = nearest_ball(robot_pos, all_balls)
            if self.target is None:
                print("[Controller] No balls found — done.")
                self._transition(State.DONE)
                return "STOP"
            print(f"[Controller] Target: {self.target}")
            self._transition(State.ALIGN)
            return "STOP"

        # --- ALIGN: turn to face target ---
        if self.state == State.ALIGN:
            error = angle_error(robot.get_angle(), angle_to_target(robot_pos, self.target))
            print(f"[Controller] Aligning — error: {error:.1f}°")
            if abs(error) <= ALIGN_THRESHOLD_DEG:
                self._transition(State.DRIVE)
                return "STOP"
            cmd = "RIGHT" if error > 0 else "LEFT"
            robot.send_command(cmd)
            return cmd

        # --- DRIVE: go to target ---
        if self.state == State.DRIVE:
            dist = distance(robot_pos, self.target)
            print(f"[Controller] Driving — {dist:.1f} cm")

            error = angle_error(robot.get_angle(), angle_to_target(robot_pos, self.target))
            if abs(error) > ALIGN_THRESHOLD_DEG * 2:
                self._transition(State.ALIGN)
                return "STOP"

            if dist <= ARRIVAL_THRESHOLD_CM:
                print("[Controller] Ball collected.")
                self.target = None
                self._transition(State.IDLE)
                return "STOP"

            robot.send_command("FORWARD")
            return "FORWARD"

        return "STOP"

    def _transition(self, new_state):
        print(f"[Controller] {self.state.name} → {new_state.name}")
        self.state = new_state
