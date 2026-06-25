"""
test_state_machine_reverse_goal_integration.py - tests reverse, goal, release, and done states.
Mocks EV3 calls so end-of-run decisions can be tested offline.
"""

import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.state_machine import GolfBotController, State
from config import GOAL_POSITION_CM, MARKER_TO_CLAW_CM
from test.world_state_helpers import world_state


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
        controller._driver.reverse = lambda rotations: calls.append(rotations)
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        self.assertIs(controller.update(world), Command.BACKWARD)
        self.assertTrue(calls)
        self.assertTrue(controller._has_reversed)

        self.assertIs(controller.update(world), Command.STOP)
        self.assertIs(controller.state, State.REVERSE_ORANGE)
        self.assertFalse(controller._has_reversed)

    def test_reverse_state_returns_to_seek_when_ball_is_found_after_reverse(self):
        controller = GolfBotController()
        controller.state = State.REVERSE_WHITE
        controller._has_reversed = True
        world = world_state(
            robot=(20.0, 60.0),
            robot_px=(100, 300),
            robot_angle=0.0,
            white_balls=[(30.0, 60.0, "open")],
        )

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_reverse_orange_goes_to_goal_when_orange_is_not_found(self):
        controller = GolfBotController()
        controller.state = State.REVERSE_ORANGE
        controller._has_reversed = True
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0, ob=None)

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.DRIVE_GOAL)

    def test_drive_goal_uses_driver_when_waypoints_are_done_and_heading_is_correct(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        controller._goal_waypoints = []
        controller._goal_approach_angle = 180.0
        controller._driver.drive_toward = lambda pose, px, arrive: (Command.FORWARD, False)
        world = world_state(robot=(100.0, 60.0), robot_px=(500, 300), robot_angle=180.0)

        command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertTrue(calls)
        self.assertIs(controller.state, State.DRIVE_GOAL)

    def test_drive_goal_turns_when_final_heading_is_wrong(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        controller._goal_waypoints = []
        controller._goal_approach_angle = 180.0
        controller._driver.turn = lambda pose, rotations, direction: calls.append((rotations, direction))
        world = world_state(robot=(100.0, 60.0), robot_px=(500, 300), robot_angle=270.0)

        command = controller.update(world)

        self.assertIs(command, Command.RIGHT)
        self.assertEqual(calls[0][1], Command.RIGHT)

    def test_drive_goal_reverses_when_heading_correction_gets_stuck(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        controller._goal_waypoints = []
        controller._goal_heading_corrections = GOAL_HEADING_MAX_CORRECTIONS
        controller._driver.reverse = lambda rotations: calls.append(rotations)
        world = world_state(robot=(100.0, 60.0), robot_px=(500, 300), robot_angle=170.0)

        command = controller.update(world)

        self.assertIs(command, Command.BACKWARD)
        self.assertTrue(calls)
        self.assertIsNone(controller._goal_waypoints)
        self.assertEqual(controller._goal_heading_corrections, 0)

    def test_drive_goal_reverses_when_release_lane_rebuild_gets_stuck(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        controller._goal_waypoints = []
        controller._goal_lane_rejections = GOAL_LANE_MAX_REJECTIONS
        controller._driver.reverse = lambda rotations: calls.append(rotations)
        world = world_state(
            robot=(GOAL_RELEASE_MARKER_PX[0] * 169.0 / 900, 65.0),
            robot_px=(GOAL_RELEASE_MARKER_PX[0], GOAL_RELEASE_MARKER_PX[1] + 20),
            robot_angle=0.0,
        )

        command = controller.update(world)

        self.assertIs(command, Command.BACKWARD)
        self.assertTrue(calls)
        self.assertIsNone(controller._goal_waypoints)
        self.assertEqual(controller._goal_lane_rejections, 0)

    def test_drive_goal_transitions_to_release_when_robot_is_at_goal(self):
        controller = GolfBotController()
        controller.state = State.DRIVE_GOAL
        controller._goal_waypoints = []
        controller._goal_approach_angle = 180.0
        controller._driver.drive_toward = lambda pose, px, arrive: (Command.STOP, True)
        world = world_state(
            robot=(GOAL_POSITION_CM[0] + MARKER_TO_CLAW_CM, GOAL_POSITION_CM[1]),
            robot_px=(50, 300),
            robot_angle=180.0,
        )

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.RELEASE)

    def test_release_opens_and_closes_gate_then_rescans(self):
        calls = []
        controller = GolfBotController()
        controller.state = State.RELEASE
        world = world_state(robot=(180.0, 60.0), robot_px=(900, 300), robot_angle=0.0)

        with patch("controller.state_machine.robot.gate_open", lambda: calls.append("open")), \
             patch("controller.state_machine.robot.gate_close", lambda: calls.append("close")), \
             patch("controller.state_machine.time.sleep"):
            command = controller.update(world)

        self.assertIs(command, Command.RELEASE)
        self.assertEqual(calls, ["open", "close"])
        self.assertIs(controller.state, State.REVERSE_WHITE)
        self.assertTrue(controller._delivered)
        self.assertFalse(controller._has_reversed)

    def test_reverse_orange_finishes_when_empty_after_a_delivery(self):
        controller = GolfBotController()
        controller.state = State.REVERSE_ORANGE
        controller._has_reversed = True
        controller._delivered = True
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0, ob=None)

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.DONE)

    def test_done_state_returns_stop(self):
        controller = GolfBotController()
        controller.state = State.DONE
        world = world_state(robot=(180.0, 60.0), robot_px=(900, 300), robot_angle=0.0)

        self.assertIs(controller.update(world), Command.STOP)


if __name__ == "__main__":
    unittest.main()
