"""
test_route_manager_integration.py - integration test for world dict -> route target.
Mocks the TSP dependency so ball priority can be tested without networkx.
"""

import sys
import types
import unittest
from unittest.mock import patch

fake_tsp = types.ModuleType("controller.tsp_christofides")
fake_tsp.christofides_route = lambda points: list(range(len(points)))
sys.modules.setdefault("controller.tsp_christofides", fake_tsp)

from controller.route_manager import RouteManager


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


if __name__ == "__main__":
    unittest.main()
