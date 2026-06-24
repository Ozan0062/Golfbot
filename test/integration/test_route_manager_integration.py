"""
test_route_manager_integration.py - integration test for world dict -> route target.
Tests nearest-ball route selection without camera, YOLO, networkx, or robot hardware.
"""

import unittest
from unittest.mock import patch

from controller.route_manager import RouteManager, _dist
from test.world_state_helpers import world_state


class RouteManagerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()

    def tearDown(self):
        self._print_patcher.stop()

    def test_route_manager_targets_white_balls_before_orange_ball(self):
        manager = RouteManager()
        robot_pos = (20.0, 20.0)
        robot_px = (100, 100)

        world_with_whites = world_state(
            white_balls=[(30.0, 20.0, "open"), (80.0, 80.0, "open"), (120.0, 40.0, "open")],
            white_balls_px=[(150, 100, "open"), (400, 400, "open"), (600, 200, "open")],
            ob=(170.0, 100.0, "open"),
            ob_px=(850, 500, "open"),
        )

        target = manager.get_target(robot_pos, robot_px, world_with_whites)
        self.assertIn(target.cm, [(30.0, 20.0), (80.0, 80.0), (120.0, 40.0)])
        self.assertNotEqual(target.cm, world_with_whites.ob[:2])

        manager.clear()
        world_with_only_orange = world_state(
            white_balls=[],
            white_balls_px=[],
            ob=(170.0, 100.0, "open"),
            ob_px=(850, 500, "open"),
        )

        target = manager.get_target(robot_pos, robot_px, world_with_only_orange)
        self.assertEqual(target.cm, (170.0, 100.0))
        self.assertEqual(target.px, (850, 500))

    def test_route_manager_selects_nearest_white_ball_by_pixel_distance(self):
        manager = RouteManager()
        world = world_state(
            white_balls=[(30.0, 20.0, "open"), (80.0, 80.0, "open"), (120.0, 40.0, "open")],
            white_balls_px=[(600, 500, "open"), (130, 100, "open"), (500, 200, "open")],
            ob=(170.0, 100.0, "open"),
            ob_px=(850, 500, "open"),
        )

        target = manager.get_target((20.0, 20.0), (100, 100), world)

        self.assertEqual(target.cm, (80.0, 80.0))
        self.assertEqual(target.px, (130, 100))
        self.assertAlmostEqual(target.dist_px, 30.0)

    def test_route_manager_re_evaluates_nearest_ball_each_call(self):
        manager = RouteManager()
        world = world_state(
            white_balls=[(30.0, 20.0, "open"), (80.0, 80.0, "open")],
            white_balls_px=[(150, 100, "open"), (800, 500, "open")],
            ob=None,
            ob_px=None,
        )

        first_target = manager.get_target((20.0, 20.0), (100, 100), world)
        second_target = manager.get_target((20.0, 20.0), (900, 500), world)

        self.assertEqual(first_target.cm, (30.0, 20.0))
        self.assertEqual(second_target.cm, (80.0, 80.0))

    def test_route_manager_orange_target_distance_uses_robot_pixels(self):
        manager = RouteManager()
        world = world_state(
            white_balls=[],
            white_balls_px=[],
            ob=(170.0, 100.0, "open"),
            ob_px=(130, 140, "open"),
        )

        target = manager.get_target((20.0, 20.0), (100, 100), world)

        self.assertEqual(target.cm, (170.0, 100.0))
        self.assertEqual(target.px, (130, 140))
        self.assertAlmostEqual(target.dist_px, 50.0)

    def test_route_manager_returns_none_when_no_balls_are_visible(self):
        manager = RouteManager()
        world = world_state(white_balls=[], white_balls_px=[], ob=None, ob_px=None)

        target = manager.get_target((20.0, 20.0), (100, 100), world)

        self.assertIsNone(target)

    def test_route_manager_uses_first_nearest_path_step_as_target(self):
        manager = RouteManager()
        path = [{"pos_cm": (40.0, 20.0), "pos_px": (220, 100), "type": "open"}]

        target = manager.get_target_nearest(path, robot_px=(100, 100), world=world_state())

        self.assertEqual(target.cm, (40.0, 20.0))
        self.assertEqual(target.px, (220, 100))
        self.assertAlmostEqual(target.dist_px, 120.0)

    def test_route_manager_returns_none_for_empty_nearest_path(self):
        manager = RouteManager()

        self.assertIsNone(manager.get_target_nearest([], robot_px=(100, 100), world=world_state()))

    def test_dist_returns_euclidean_distance(self):
        self.assertAlmostEqual(_dist((0, 0), (3, 4)), 5.0)


if __name__ == "__main__":
    unittest.main()
