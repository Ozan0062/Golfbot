"""
test_state_machine_seek_avoid_integration.py - tests SEEK and AVOID controller decisions.
Mocks routing and EV3 movement so obstacle and waypoint behaviour runs offline.
"""

import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.route_manager import RouteTarget
from controller.state_machine import GolfBotController, State
from test.world_state_helpers import world_state


class StateMachineSeekAvoidTests(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()

    def tearDown(self):
        self._print_patcher.stop()

    def _controller_with_target(self, target):
        controller = GolfBotController()
        controller._route.get_target_nearest = lambda path, robot_px, world: target
        return controller

    def test_seek_uses_avoid_state_when_cross_blocks_path(self):
        controller = self._controller_with_target(
            RouteTarget(cm=(100.0, 20.0), px=(500, 100), dist_px=400.0)
        )
        world = world_state(
            robot=(20.0, 20.0),
            robot_px=(100, 100),
            robot_angle=0.0,
            cross_px=(300, 100),
        )

        with patch("controller.state_machine.find_nearest", return_value={"pos_px": (500, 100)}):
            command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.AVOID)
        self.assertIsNotNone(controller._avoid_target)

    def test_seek_open_field_ball_goes_to_approach(self):
        controller = self._controller_with_target(
            RouteTarget(cm=(100.0, 60.0), px=(500, 300), dist_px=400.0)
        )
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        with patch("controller.state_machine.find_nearest", return_value={"pos_px": (500, 300)}):
            command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.APPROACH)

    def test_ball_at_the_cross_is_collected_with_staged_approach(self):
        controller = self._controller_with_target(
            RouteTarget(cm=(85.0, 75.0), px=(470, 320), dist_px=430.0)
        )
        world = world_state(
            robot=(20.0, 20.0),
            robot_px=(100, 100),
            robot_angle=0.0,
            cross_px=(450, 300),
        )

        with patch("controller.state_machine.find_nearest", return_value={"pos_px": (470, 320)}):
            command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.AVOID)
        self.assertTrue(controller._is_wall_ball)
        self.assertIsNotNone(controller._corner_approach_angle)

    def test_state_machine_stops_when_robot_pose_is_missing(self):
        controller = GolfBotController()
        world = world_state(robot=None, robot_px=None, robot_angle=None)

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_avoid_without_waypoint_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = None
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)

    def test_avoid_reached_normal_waypoint_returns_to_seek(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (105, 305)
        controller._driver.drive_toward = lambda pose, wp, arrive: (Command.STOP, True)
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.SEEK)
        self.assertIsNone(controller._avoid_target)

    def test_avoid_advances_to_next_corner_waypoint(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (105, 305)
        controller._corner_waypoints = [(200, 300)]
        controller._driver.drive_toward = lambda pose, wp, arrive: (Command.STOP, True)
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=40.0)

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
        controller._driver.drive_toward = lambda pose, wp, arrive: (Command.STOP, True)
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=45.0)

        command = controller.update(world)

        self.assertIs(command, Command.STOP)
        self.assertIs(controller.state, State.APPROACH)
        self.assertIsNone(controller._avoid_target)

    def test_avoid_returns_driver_command_while_waypoint_is_not_reached(self):
        controller = GolfBotController()
        controller.state = State.AVOID
        controller._avoid_target = (300, 300)
        controller._driver.drive_toward = lambda pose, wp, arrive: (Command.FORWARD, False)
        world = world_state(robot=(20.0, 60.0), robot_px=(100, 300), robot_angle=0.0)

        command = controller.update(world)

        self.assertIs(command, Command.FORWARD)
        self.assertIs(controller.state, State.AVOID)


if __name__ == "__main__":
    unittest.main()
