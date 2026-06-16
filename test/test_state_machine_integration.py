"""
test_state_machine_integration.py - integration tests for GolfBot controller decisions.
Mocks route planning and EV3 calls so the state machine runs without the robot.
"""

import sys
import types
import unittest
from unittest.mock import patch

fake_tsp = types.ModuleType("controller.tsp_christofides")
fake_tsp.christofides_route = lambda points: list(range(len(points)))
sys.modules.setdefault("controller.tsp_christofides", fake_tsp)

from controller.commands import Command
from controller.state_machine import GolfBotController, State


class StateMachineIntegrationTests(unittest.TestCase):
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

    def test_state_machine_sends_drive_command_to_mocked_ev3(self):
        calls = []

        def fake_drive(rotations):
            calls.append(("drive", rotations))

        with patch("controller.state_machine.robot.drive", fake_drive):
            controller = GolfBotController()
            world = {
                "robot": (20.0, 60.0),
                "robot_px": (100, 300),
                "robot_angle": 0.0,
                "white_balls": [(100.0, 60.0)],
                "white_balls_px": [(500, 300)],
                "ob": None,
                "ob_px": None,
                "cross_px": None,
            }

            self.assertIs(controller.update(world), Command.STOP)
            self.assertIs(controller.state, State.ALIGN)

            self.assertIs(controller.update(world), Command.STOP)
            self.assertIs(controller.state, State.APPROACH)

            self.assertIs(controller.update(world), Command.FORWARD)
            self.assertIs(controller.state, State.ALIGN)

        self.assertTrue(calls)
        self.assertEqual(calls[0][0], "drive")
        self.assertGreater(calls[0][1], 0)


if __name__ == "__main__":
    unittest.main()
