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
from controller.route_manager import RouteTarget
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

    def test_align_turns_when_heading_error_is_large(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.ALIGN
        controller._locked_target = RouteTarget(
            cm=(20.0, 100.0),
            px=(100, 500),
            dist_px=200.0,
        )
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
        }

        def fake_turn(rotations, direction):
            calls.append((rotations, direction))

        with patch("controller.state_machine.robot.turn", fake_turn):
            command = controller.update(world)

        self.assertIs(command, Command.RIGHT)
        self.assertEqual(calls[0][1], "RIGHT")
        self.assertGreater(calls[0][0], 0)

    def test_approach_collects_when_target_is_within_collect_radius(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(
            cm=(20.0, 60.0),
            px=(110, 305),
            dist_px=12.0,
        )
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
        }

        with patch("controller.state_machine.robot.collect", lambda: calls.append("collect")):
            command = controller.update(world)

        self.assertIs(command, Command.COLLECT)
        self.assertEqual(calls, ["collect"])
        self.assertIs(controller.state, State.SEEK)
        self.assertIsNone(controller._locked_target)

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

    def test_drive_goal_drives_forward_when_goal_is_far_and_aligned(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        world = {
            "robot": (20.0, 60.0),
            "robot_px": (100, 300),
            "robot_angle": 0.0,
        }

        with patch("controller.state_machine.robot.drive", lambda rotations: calls.append(rotations)):
            command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertTrue(calls)
        self.assertGreater(calls[0], 0)

    def test_drive_goal_transitions_to_release_when_robot_is_at_goal(self):
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        world = {
            "robot": (180.0, 60.0),
            "robot_px": (900, 300),
            "robot_angle": 0.0,
        }

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.RELEASE)

    def test_release_opens_and_closes_gate_then_marks_done(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.RELEASE
        world = {
            "robot": (180.0, 60.0),
            "robot_px": (900, 300),
            "robot_angle": 0.0,
        }

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
        world = {
            "robot": (180.0, 60.0),
            "robot_px": (900, 300),
            "robot_angle": 0.0,
        }

        self.assertIs(controller.update(world), Command.STOP)


if __name__ == "__main__":
    unittest.main()
