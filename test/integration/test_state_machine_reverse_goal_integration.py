"""
test_state_machine_reverse_goal_integration.py - tests reverse, goal, release, and done states.
Mocks EV3 calls so end-of-run decisions can be tested offline.
"""

import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.state_machine import GolfBotController, State


class StateMachineReverseGoalTests(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()

    def tearDown(self):
        self._print_patcher.stop()

    def test_reverse_state_reverses_first_then_moves_to_next_state_when_empty(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.REVERSE_WHITE
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
            "white_balls": [],
            "ob": None,
        }

        with patch("controller.state_machine.robot.reverse", lambda rotations: calls.append(rotations)):
            self.assertIs(controller.update(world), Command.BACKWARD)

        self.assertTrue(calls)
        self.assertTrue(controller._has_reversed)

        controller._pose._valid_after = 0
        self.assertIs(controller.update(world), Command.STOP)

        self.assertIs(controller.state, State.REVERSE_ORANGE)
        self.assertFalse(controller._has_reversed)

    def test_reverse_state_returns_to_seek_when_ball_is_found_after_reverse(self):
        controller = GolfBotController()
        controller.state = State.REVERSE_WHITE
        controller._has_reversed = True
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
            "white_balls": [(30.0, 60.0)],
            "ob": None,
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_reverse_orange_goes_to_goal_when_orange_is_not_found(self):
        controller = GolfBotController()
        controller.state = State.REVERSE_ORANGE
        controller._has_reversed = True
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0, "ob": None}

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.DRIVE_GOAL)

    def test_drive_goal_drives_forward_when_goal_is_far_and_aligned(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        with patch("controller.state_machine.robot.drive", lambda rotations: calls.append(rotations)):
            command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertTrue(calls)
        self.assertGreater(calls[0], 0)

    def test_drive_goal_turns_when_not_aligned(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 90.0}

        with patch("controller.state_machine.robot.turn", lambda rotations, direction: calls.append((rotations, direction))):
            command = controller.update(world)

        self.assertIs(command, Command.LEFT)
        self.assertEqual(calls[0][1], "LEFT")

    def test_drive_goal_transitions_to_release_when_robot_is_at_goal(self):
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        world = {"robot": (180.0, 60.0), "robot_px": (900, 300), "robot_angle": 0.0}

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.RELEASE)

    def test_release_opens_and_closes_gate_then_marks_done(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.RELEASE
        world = {"robot": (180.0, 60.0), "robot_px": (900, 300), "robot_angle": 0.0}

        with patch("controller.state_machine.robot.gate_open", lambda: calls.append("open")), patch(
            "controller.state_machine.robot.gate_close", lambda: calls.append("close")
        ), patch("controller.state_machine.time.sleep"):
            command = controller.update(world)

        self.assertIs(command, Command.RELEASE)
        self.assertEqual(calls, ["open", "close"])
        self.assertIs(controller.state, State.DONE)

    def test_done_state_returns_stop(self):
        controller = GolfBotController()
        controller.state = State.DONE
        world = {"robot": (180.0, 60.0), "robot_px": (900, 300), "robot_angle": 0.0}

        self.assertIs(controller.update(world), Command.STOP)


if __name__ == "__main__":
    unittest.main()

