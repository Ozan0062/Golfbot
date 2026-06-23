"""
test_navigation_unit.py - offline unit tests for GolfBot navigation math.
Run with unittest or pytest; no camera or EV3 required.
"""

import math
import unittest

from controller.navigation import (
    _point_to_segment_dist,
    angle_error,
    angle_to_target,
    classify_zone,
    cm_to_pixels,
    cross_approach_angle,
    cross_trigger_radius,
    obstacle_waypoint,
    path_is_clear,
    staging_point,
    wall_approach_angle,
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

    def test_cm_to_pixels_scales_field_coordinates_to_warped_image(self):
        self.assertEqual(cm_to_pixels((90.0, 60.0), 900, 600, 180.0, 120.0), (450.0, 300.0))

    def test_wall_approach_angle_returns_wall_corner_and_open_angles(self):
        self.assertEqual(wall_approach_angle(["top"]), -90.0)
        self.assertEqual(wall_approach_angle(["bottom"]), 90.0)
        self.assertEqual(wall_approach_angle(["left"]), 180.0)
        self.assertEqual(wall_approach_angle(["right"]), 0.0)
        self.assertEqual(wall_approach_angle(["top", "left"]), -135.0)
        self.assertEqual(wall_approach_angle(["bottom", "right"]), 45.0)
        self.assertIsNone(wall_approach_angle([]))

    def test_staging_point_is_placed_behind_target_on_approach_axis(self):
        self.assertAlmostEqual(staging_point((100, 100), 0.0, 40)[0], 60.0)
        self.assertAlmostEqual(staging_point((100, 100), 0.0, 40)[1], 100.0)
        self.assertAlmostEqual(staging_point((100, 100), -90.0, 40)[0], 100.0)
        self.assertAlmostEqual(staging_point((100, 100), -90.0, 40)[1], 140.0)

    def test_point_to_segment_distance_handles_zero_length_segment(self):
        self.assertAlmostEqual(_point_to_segment_dist((3, 4), (0, 0), (0, 0)), 5.0)

    def test_obstacle_waypoint_returns_none_for_zero_length_path(self):
        self.assertIsNone(obstacle_waypoint((100, 100), (100, 100), (120, 120), 70, 900, 600))


class CrossApproachTests(unittest.TestCase):
    CROSS = (450, 300)   # cross centre

    def test_cross_approach_angle_picks_diagonal_toward_centre_per_quadrant(self):
        # Heading always points from the ball back toward the cross centre,
        # quantised to the nearest 45deg diagonal.
        self.assertEqual(cross_approach_angle((500, 350), self.CROSS), -135.0)  # down-right -> up-left
        self.assertEqual(cross_approach_angle((500, 250), self.CROSS),  135.0)  # up-right   -> down-left
        self.assertEqual(cross_approach_angle((400, 350), self.CROSS),  -45.0)  # down-left  -> up-right
        self.assertEqual(cross_approach_angle((400, 250), self.CROSS),   45.0)  # up-left    -> down-right

    def test_cross_approach_staging_point_sits_radially_outside_the_ball(self):
        # The staging point must be further from the cross than the ball, so the
        # robot drives inward and the cross stays behind the ball.
        ball = (500, 350)
        angle = cross_approach_angle(ball, self.CROSS)
        sp = staging_point(ball, angle, 100)
        ball_d = math.hypot(ball[0] - self.CROSS[0], ball[1] - self.CROSS[1])
        sp_d = math.hypot(sp[0] - self.CROSS[0], sp[1] - self.CROSS[1])
        self.assertGreater(sp_d, ball_d)

    def test_cross_trigger_radius_uses_detected_size_when_available(self):
        # half of the larger box side + clearance
        self.assertAlmostEqual(cross_trigger_radius((100, 80), 999, 60), 50 + 60)

    def test_cross_trigger_radius_falls_back_when_size_missing(self):
        self.assertAlmostEqual(cross_trigger_radius(None, 53, 60), 53 + 60)


if __name__ == "__main__":
    unittest.main()
