"""
test_dijkstras_unit.py - offline unit tests for route graph planning.
Uses synthetic WorldState objects, so no camera, YOLO model, or robot is needed.
"""

import unittest

from controller.dijkstras import (
    calculate_best_route,
    create_nodes_and_edges,
    get_path,
    line_intersects_obstacle,
)
from test.world_state_helpers import world_state


class DijkstrasUnitTests(unittest.TestCase):
    def test_line_intersects_obstacle_only_when_cross_is_on_forward_segment(self):
        self.assertTrue(
            line_intersects_obstacle((0, 0), (10, 0), (5, 1), clearance=2)
        )
        self.assertFalse(
            line_intersects_obstacle((0, 0), (10, 0), (-5, 0), clearance=2)
        )
        self.assertFalse(
            line_intersects_obstacle((0, 0), (10, 0), (5, 5), clearance=2)
        )

    def test_create_nodes_and_edges_includes_robot_balls_orange_and_goal(self):
        world = world_state(
            robot=(20.0, 20.0),
            robot_px=(100, 100),
            robot_angle=0.0,
            white_balls=[(40.0, 20.0, "open")],
            white_balls_px=[(220, 100, "open")],
            white_wall_balls=[(80.0, 20.0, "wall")],
            white_wall_balls_px=[(440, 100, "wall")],
            white_corner_balls=[(120.0, 20.0, "corner")],
            white_corner_balls_px=[(660, 100, "corner")],
            ob=(150.0, 80.0, "open"),
            ob_px=(800, 400, "open"),
        )

        graph = create_nodes_and_edges(world)

        self.assertTrue(graph.has_node("robot"))
        self.assertTrue(graph.has_node("goal"))
        self.assertTrue(graph.has_node("ob"))
        self.assertEqual(len([node for node in graph.nodes if str(node).startswith("wb_")]), 3)
        self.assertGreater(graph.number_of_edges(), graph.number_of_nodes())

    def test_calculate_best_route_returns_white_balls_then_orange(self):
        world = world_state(
            robot=(20.0, 20.0),
            robot_px=(100, 100),
            robot_angle=0.0,
            white_balls=[(40.0, 20.0, "open"), (80.0, 20.0, "open")],
            white_balls_px=[(220, 100, "open"), (440, 100, "open")],
            ob=(150.0, 80.0, "open"),
            ob_px=(800, 400, "open"),
        )

        path = calculate_best_route(create_nodes_and_edges(world))

        self.assertEqual([step["id"] for step in path][-1], "ob")
        self.assertTrue(all(step["id"].startswith("wb_") for step in path[:-1]))

    def test_get_path_returns_empty_when_robot_is_missing(self):
        world = world_state(
            white_balls=[(40.0, 20.0, "open")],
            white_balls_px=[(220, 100, "open")],
        )

        self.assertEqual(get_path(world), [])


if __name__ == "__main__":
    unittest.main()
