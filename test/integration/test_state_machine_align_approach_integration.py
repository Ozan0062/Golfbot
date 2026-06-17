"""
test_state_machine_align_approach_integration.py - tests the merged APPROACH state.

APPROACH now turns to face the ball and drives toward it (one step per frame),
then grabs it once within reach. Motor/claw calls are mocked so this runs offline.
Fixtures keep cm == 0.2 * px (the real warped→cm scale) so headings are consistent.
"""

import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.route_manager import RouteTarget
from controller.state_machine import GolfBotController, State
from config import COLLECT_RADIUS_PX


class StateMachineApproachTests(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()

    def tearDown(self):
        self._print_patcher.stop()

    def test_seek_open_field_ball_goes_to_approach(self):
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

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.APPROACH)

    def test_approach_drives_toward_far_aligned_ball(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(cm=(80.0, 60.0), px=(400, 300), dist_px=300.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        with patch("controller.ev3_controller.drive", lambda rotations: calls.append(rotations)):
            command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertTrue(calls)
        self.assertGreater(calls[0], 0)
        self.assertIs(controller.state, State.APPROACH)

    def test_approach_turns_toward_misaligned_ball(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(cm=(80.0, 60.0), px=(400, 300), dist_px=300.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 90.0}

        with patch("controller.ev3_controller.turn",
                   lambda rotations, direction: calls.append((rotations, direction))):
            command = controller.update(world)

        self.assertIs(command, Command.LEFT)
        self.assertEqual(calls[0][1], "LEFT")
        self.assertGreater(calls[0][0], 0)

    def test_approach_without_target_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = None
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_approach_collects_when_within_radius_and_aligned(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(cm=(30.0, 60.0), px=(150, 300), dist_px=50.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        with patch("controller.ev3_controller.collect", lambda: calls.append("collect")):
            command = controller.update(world)

        self.assertIs(command, Command.COLLECT)
        self.assertEqual(calls, ["collect"])
        self.assertIs(controller.state, State.SEEK)
        self.assertIsNone(controller._locked_target)

    def test_approach_within_radius_but_off_heading_turns_in_place(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._locked_target = RouteTarget(cm=(30.0, 60.0), px=(150, 300), dist_px=50.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 90.0}

        with patch("controller.ev3_controller.turn",
                   lambda rotations, direction: calls.append((rotations, direction))), \
             patch("controller.ev3_controller.collect", lambda: calls.append("collect")):
            command = controller.update(world)

        self.assertIn(command, (Command.LEFT, Command.RIGHT))
        self.assertNotIn("collect", calls)
        self.assertIs(controller.state, State.APPROACH)

    def test_approach_wall_ball_collects_then_reverses(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.APPROACH
        controller._is_wall_ball = True
        controller._locked_target = RouteTarget(cm=(30.0, 60.0), px=(150, 300), dist_px=50.0)
        world = {"robot": (20.0, 60.0), "robot_px": (100, 300), "robot_angle": 0.0}

        with patch("controller.ev3_controller.collect", lambda: calls.append("collect")), \
             patch("controller.ev3_controller.reverse", lambda rotations: calls.append(("reverse", rotations))):
            command = controller.update(world)

        self.assertIs(command, Command.COLLECT)
        self.assertIn("collect", calls)
        self.assertEqual(calls[-1][0], "reverse")
        self.assertFalse(controller._is_wall_ball)


if __name__ == "__main__":
    unittest.main()
