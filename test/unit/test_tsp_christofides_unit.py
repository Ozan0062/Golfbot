"""
test_tsp_christofides_unit.py - offline unit tests for Christofides route helper.
Verifies edge cases and basic route properties without robot hardware.
"""

import unittest

from controller.tsp_christofides import _dist, christofides_route


class TspChristofidesUnitTests(unittest.TestCase):
    def test_route_handles_empty_single_and_two_point_inputs(self):
        self.assertEqual(christofides_route([]), [])
        self.assertEqual(christofides_route([(0, 0)]), [0])
        self.assertEqual(christofides_route([(0, 0), (1, 1)]), [0, 1])

    def test_route_starts_at_robot_and_visits_each_point_once(self):
        points = [(0, 0), (1, 0), (0, 1), (1, 1)]

        route = christofides_route(points)

        self.assertEqual(route[0], 0)
        self.assertEqual(sorted(route), [0, 1, 2, 3])

    def test_dist_returns_euclidean_distance(self):
        self.assertAlmostEqual(_dist((0, 0), (3, 4)), 5.0)


if __name__ == "__main__":
    unittest.main()