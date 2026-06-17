"""
test_route_manager_integration.py - integration test for world dict -> route target.
Tests nearest-ball route selection without camera, YOLO, networkx, or robot hardware.
"""

import unittest
from unittest.mock import patch

from controller.route_manager import RouteManager, _dist


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

        world_with_whites = {
            "white_balls": [(30.0, 20.0), (80.0, 80.0), (120.0, 40.0)],
            "white_balls_px": [(150, 100), (400, 400), (600, 200)],
            "ob": (170.0, 100.0),
            "ob_px": (850, 500),
        }

        target = manager.get_target(robot_pos, robot_px, world_with_whites)
        self.assertIn(target.cm, world_with_whites["white_balls"])
        self.assertNotEqual(target.cm, world_with_whites["ob"])

        manager.clear()
        world_with_only_orange = {
            "white_balls": [],
            "white_balls_px": [],
            "ob": (170.0, 100.0),
            "ob_px": (850, 500),
        }

        target = manager.get_target(robot_pos, robot_px, world_with_only_orange)
        self.assertEqual(target.cm, (170.0, 100.0))
        self.assertEqual(target.px, (850, 500))

    def test_route_manager_selects_nearest_white_ball_by_pixel_distance(self):
        manager = RouteManager()
        world = {
            "white_balls": [(30.0, 20.0), (80.0, 80.0), (120.0, 40.0)],
            "white_balls_px": [(600, 500), (130, 100), (500, 200)],
            "ob": (170.0, 100.0),
            "ob_px": (850, 500),
        }

        target = manager.get_target((20.0, 20.0), (100, 100), world)

        self.assertEqual(target.cm, (80.0, 80.0))
        self.assertEqual(target.px, (130, 100))
        self.assertAlmostEqual(target.dist_px, 30.0)

    def test_route_manager_re_evaluates_nearest_ball_each_call(self):
        manager = RouteManager()
        world = {
            "white_balls": [(30.0, 20.0), (80.0, 80.0)],
            "white_balls_px": [(150, 100), (800, 500)],
            "ob": None,
            "ob_px": None,
        }

        first_target = manager.get_target((20.0, 20.0), (100, 100), world)
        second_target = manager.get_target((20.0, 20.0), (900, 500), world)

        self.assertEqual(first_target.cm, (30.0, 20.0))
        self.assertEqual(second_target.cm, (80.0, 80.0))

    def test_route_manager_orange_target_distance_uses_robot_pixels(self):
        manager = RouteManager()
        world = {
            "white_balls": [],
            "white_balls_px": [],
            "ob": (170.0, 100.0),
            "ob_px": (130, 140),
        }

        target = manager.get_target((20.0, 20.0), (100, 100), world)

        self.assertEqual(target.cm, (170.0, 100.0))
        self.assertEqual(target.px, (130, 140))
        self.assertAlmostEqual(target.dist_px, 50.0)

    def test_route_manager_returns_none_when_no_balls_are_visible(self):
        manager = RouteManager()
        world = {
            "white_balls": [],
            "white_balls_px": [],
            "ob": None,
            "ob_px": None,
        }

        target = manager.get_target((20.0, 20.0), (100, 100), world)

        self.assertIsNone(target)

    def test_dist_returns_euclidean_distance(self):
        self.assertAlmostEqual(_dist((0, 0), (3, 4)), 5.0)


if __name__ == "__main__":
    unittest.main()
