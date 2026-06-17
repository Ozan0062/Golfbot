"""
test_navigation_unit.py - offline unit tests for GolfBot navigation math.
Run with unittest or pytest; no camera or EV3 required.
"""

import math
import unittest

from controller.navigation import (
    angle_error,
    angle_to_target,
    classify_zone,
    obstacle_waypoint,
    path_is_clear,
)


class NavigationUnitTests(unittest.TestCase):
    def test_angle_math_turns_towards_ball_by_shortest_direction(self):
        robot_pos = (50.0, 60.0)
        ball_pos = (100.0, 60.0)

        desired_angle = angle_to_target(robot_pos, ball_pos)

        self.assertAlmostEqual(desired_angle, 0.0)
        self.assertAlmostEqual(angle_error(350.0, 10.0), 20.0)
        self.assertAlmostEqual(angle_error(90.0, desired_angle), -90.0)

    def test_zone_classification_finds_open_wall_and_corner_balls(self):
        width = 900
        height = 600
        margin = 120

        self.assertEqual(classify_zone((450, 300), margin, width, height), ("open", []))
        self.assertEqual(classify_zone((50, 300), margin, width, height), ("wall", ["left"]))
        self.assertEqual(
            classify_zone((50, 50), margin, width, height),
            ("corner", ["left", "top"]),
        )

    def test_path_clearance_detects_cross_blocking_the_robot_path(self):
        start_px = (100, 100)
        target_px = (500, 100)
        blocking_cross = (300, 105)
        harmless_cross = (300, 250)

        clear, blocker = path_is_clear(start_px, target_px, [blocking_cross], 70)
        self.assertFalse(clear)
        self.assertEqual(blocker, blocking_cross)

        clear, blocker = path_is_clear(start_px, target_px, [harmless_cross], 70)
        self.assertTrue(clear)
        self.assertIsNone(blocker)

    def test_obstacle_waypoint_is_inside_field_and_away_from_blocker(self):
        robot_px = (100, 100)
        target_px = (500, 100)
        cross_px = (300, 105)
        clearance = 140

        waypoint = obstacle_waypoint(robot_px, target_px, cross_px, clearance, 900, 600)

        self.assertIsNotNone(waypoint)
        self.assertGreaterEqual(waypoint[0], clearance)
        self.assertLessEqual(waypoint[0], 900 - clearance)
        self.assertGreaterEqual(waypoint[1], clearance)
        self.assertLessEqual(waypoint[1], 600 - clearance)
        self.assertGreater(math.dist(waypoint, cross_px), 70)


if __name__ == "__main__":
    unittest.main()
