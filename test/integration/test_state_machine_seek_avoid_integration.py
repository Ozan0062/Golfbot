"""
test_state_machine_seek_avoid_integration.py - tests SEEK and AVOID controller decisions.
Mocks EV3 calls so obstacle and waypoint behaviour can be tested offline.
"""

import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.state_machine import GolfBotController, State


class StateMachineSeekAvoidTests(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()

    def tearDown(self):
        self._print_patcher.stop()

    def test_state_machine_uses_avoid_state_when_cross_blocks_path(self):
        controller = GolfBotController()
        world = {
            "robot": (20.0, 20.0),
            "robot_px": (100, 100),
            "robot_angle": 0.0,
            "white_balls": [(100.0, 20.0)],
            "white_balls_px": [(500, 100)],
            "ob": None,
            "ob_px": None,
            "cross_px": (300, 100),
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.AVOID)

    def test_state_machine_stops_when_robot_pose_is_missing(self):
        controller = GolfBotController()
        world = {
            "robot": None,
            "robot_px": None,
            "robot_angle": None,
            "white_balls": [(100.0, 20.0)],
            "white_balls_px": [(500, 100)],
            "ob": None,
            "ob_px": None,
            "cross_px": None,
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_avoid_without_waypoint_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = None
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_avoid_reached_normal_waypoint_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (105, 305)
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)
        self.assertIsNone(controller._avoid_target)

    def test_avoid_advances_to_next_corner_waypoint(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (105, 305)
        controller._corner_waypoints = [(200, 300)]
        controller._corner_approach_angle = 45.0
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 40.0,
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.AVOID)
        self.assertEqual(controller._avoid_target, (200, 300))
        self.assertEqual(controller._corner_waypoints, [])

    def test_avoid_reached_last_wall_waypoint_goes_to_approach(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (105, 305)
        controller._is_wall_ball = True
        controller._corner_approach_angle = 45.0
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 40.0,
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.APPROACH)
        self.assertIsNone(controller._avoid_target)

    def test_avoid_turns_toward_waypoint_when_heading_is_wrong(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (100, 500)
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
        }

        with patch("controller.state_machine.robot.turn", lambda rotations, direction: calls.append((rotations, direction))):
            command = controller.update(world)

        self.assertIs(command, Command.RIGHT)
        self.assertEqual(calls[0][1], "RIGHT")

    def test_avoid_drives_toward_waypoint_when_aligned(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (300, 300)
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
        }

        with patch("controller.state_machine.robot.drive", lambda rotations: calls.append(rotations)):
            command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()

