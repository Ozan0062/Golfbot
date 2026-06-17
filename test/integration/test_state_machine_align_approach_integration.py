"""
test_state_machine_align_approach_integration.py - tests ALIGN and APPROACH decisions.
Mocks motor and claw calls so ball approach behaviour can be tested offline.
"""

import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.route_manager import RouteTarget
from controller.state_machine import COLLECT_RADIUS_PX, GolfBotController, State


class StateMachineAlignApproachTests(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()

    def tearDown(self):
        self._print_patcher.stop()

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
        controller._locked_target = RouteTarget(cm=(20.0, 100.0), px=(100, 500), dist_px=200.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        def fake_turn(rotations, direction):
            calls.append((rotations, direction))

        with patch("controller.state_machine.robot.turn", fake_turn):
            command = controller.update(world)

        self.assertIs(command, Command.RIGHT)
        self.assertEqual(calls[0][1], "RIGHT")
        self.assertGreater(calls[0][0], 0)

    def test_align_without_target_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.ALIGN
        controller._locked_target = None
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_approach_collects_when_target_is_within_collect_radius(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(cm=(20.0, 60.0), px=(110, 305), dist_px=12.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        with patch("controller.state_machine.robot.collect", lambda: calls.append("collect")):
            command = controller.update(world)

        self.assertIs(command, Command.COLLECT)
        self.assertEqual(calls, ["collect"])
        self.assertIs(controller.state, State.SEEK)
        self.assertIsNone(controller._locked_target)

    def test_approach_without_target_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = None
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_approach_close_ball_forces_strict_realign_when_heading_is_wrong(self):
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(
            cm=(20.0, 100.0),
            px=(100, 300 + COLLECT_RADIUS_PX - 1),
            dist_px=20.0,
        )
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.ALIGN)
        self.assertTrue(controller._strict_align)

    def test_approach_wall_ball_collects_and_reverses(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._is_wall_ball = True
        controller._locked_target = RouteTarget(cm=(20.0, 60.0), px=(110, 305), dist_px=12.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 20.0}

        with patch("controller.state_machine.robot.collect", lambda: calls.append("collect")), patch(
            "controller.state_machine.robot.reverse", lambda rotations: calls.append(("reverse", rotations))
        ):
            command = controller.update(world)

        self.assertIs(command, Command.COLLECT)
        self.assertIn("collect", calls)
        self.assertEqual(calls[-1][0], "reverse")
        self.assertFalse(controller._is_wall_ball)

    def test_approach_wall_ball_drives_small_step_without_realigning(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._is_wall_ball = True
        controller._locked_target = RouteTarget(cm=(100.0, 60.0), px=(300, 300), dist_px=200.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        with patch("controller.state_machine.robot.drive", lambda rotations: calls.append(rotations)):
            command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertIs(controller.state, State.APPROACH)
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()

