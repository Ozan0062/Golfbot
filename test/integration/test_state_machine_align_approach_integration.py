"""
test_state_machine_align_approach_integration.py - tests the APPROACH state.
Movement and claw actions are mocked so the controller can be tested offline.
"""

import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.route_manager import RouteTarget
from controller.state_machine import GolfBotController, State
from config import MARKER_TO_CLAW_CM
from test.world_state_helpers import world_state


class StateMachineApproachTests(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()

    def tearDown(self):
        self._print_patcher.stop()

    def _approach_controller(self, target_cm=(80.0, 60.0), target_px=(400, 300)):
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(cm=target_cm, px=target_px, dist_px=300.0)
        return controller

    def test_approach_returns_driver_command_for_far_ball(self):
        controller = self._approach_controller()
        controller._driver.drive_toward = lambda pose, px, arrive: (Command.FORWARD, False)
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertIs(controller.state, State.APPROACH)

    def test_approach_without_target_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = None
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_approach_collects_when_claw_is_at_ball_and_aligned(self):
        calls = []
        target_cm = (20.0 + MARKER_TO_CLAW_CM, 60.0)
        controller = self._approach_controller(target_cm=target_cm, target_px=(150, 300))
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        with patch("controller.state_machine.robot.close_claw", lambda: calls.append("close_claw")), \
             patch("controller.state_machine.robot.gate_rotate", lambda: calls.append("gate_rotate")), \
             patch("controller.state_machine.robot.reset_claw", lambda: calls.append("reset")):
            command = controller.update(world)

        self.assertIs(command, Command.COLLECT)
        self.assertEqual(calls, ["close_claw", "gate_rotate", "reset"])
        self.assertIs(controller.state, State.SEEK)
        self.assertIsNone(controller._locked_target)

    def test_approach_within_radius_but_off_heading_turns_in_place(self):
        calls = []
        target_cm = (20.0 + MARKER_TO_CLAW_CM, 64.0)
        controller = self._approach_controller(target_cm=target_cm, target_px=(150, 320))
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        controller._driver.turn = lambda pose, rotations, direction: calls.append((rotations, direction))
        with patch("controller.state_machine.robot.close_claw", lambda: calls.append("close_claw")):
            command = controller.update(world)

        self.assertIs(command, Command.RIGHT)
        self.assertEqual(calls[0][1], Command.RIGHT)
        self.assertNotIn("close_claw", calls)
        self.assertIs(controller.state, State.APPROACH)

    def test_approach_wall_ball_collects_then_reverses(self):
        calls = []
        target_cm = (20.0 + MARKER_TO_CLAW_CM, 60.0)
        controller = self._approach_controller(target_cm=target_cm, target_px=(150, 300))
        controller._is_wall_ball = True
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        controller._driver.reverse = lambda rotations: calls.append(("reverse", rotations))
        with patch("controller.state_machine.robot.close_claw", lambda: calls.append("close_claw")), \
             patch("controller.state_machine.robot.gate_rotate", lambda: calls.append("gate_rotate")), \
             patch("controller.state_machine.robot.reset_claw", lambda: calls.append("reset")):
            command = controller.update(world)

        self.assertIs(command, Command.COLLECT)
        self.assertIn("close_claw", calls)
        self.assertEqual(calls[1][0], "reverse")
        self.assertFalse(controller._is_wall_ball)


if __name__ == "__main__":
    unittest.main()
