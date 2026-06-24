"""
test_state_machine_helpers_unit.py - unit tests for motion helper functions.
Verifies distance and rotation conversion helpers without controller state.
"""

import unittest

from controller.motion import (
    angle_to_rotations,
    corner_approach_waypoints,
    distance_px,
    get_price,
    plan_route_waypoints,
    px_to_rotations,
)
from config import (
    CORNER_STAGE_DISTANCES_PX,
    FIELD_EDGE_MARGIN_PX,
    WARPED_HEIGHT,
    WARPED_WIDTH,
)


class MotionHelperTests(unittest.TestCase):
    def test_module_helpers_handle_none_and_rotation_conversion(self):
        self.assertEqual(distance_px(None, (1, 2)), 0.0)
        self.assertEqual(distance_px((1, 2), None), 0.0)
        self.assertAlmostEqual(distance_px((0, 0), (3, 4)), 5.0)
        self.assertGreater(px_to_rotations(100), 0)
        self.assertGreater(angle_to_rotations(25), 0)
        self.assertGreater(angle_to_rotations(-25), 0)

    def test_corner_approach_waypoints_are_clamped_inside_field(self):
        waypoints = corner_approach_waypoints(
            (100, 100),
            (10, 10),
            -135.0,
            CORNER_STAGE_DISTANCES_PX,
            WARPED_WIDTH,
            WARPED_HEIGHT,
            FIELD_EDGE_MARGIN_PX,
        )

        self.assertTrue(waypoints)
        for x, y in waypoints:
            self.assertGreaterEqual(x, FIELD_EDGE_MARGIN_PX)
            self.assertGreaterEqual(y, FIELD_EDGE_MARGIN_PX)

    def test_plan_route_waypoints_returns_direct_route_for_open_ball(self):
        route = plan_route_waypoints((100, 300), (500, 300), cross_px=None)

        self.assertEqual(route, [(500, 300)])

    def test_plan_route_waypoints_adds_dodge_when_cross_blocks_path(self):
        route = plan_route_waypoints((100, 300), (500, 300), cross_px=(300, 300))

        self.assertGreater(len(route), 1)
        self.assertEqual(route[-1], (500, 300))

    def test_get_price_rejects_missing_coordinates(self):
        with self.assertRaises(ValueError):
            get_price(None, (500, 300))


if __name__ == "__main__":
    unittest.main()
