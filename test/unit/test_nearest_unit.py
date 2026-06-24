"""
test_nearest_unit.py - offline unit tests for route graph planning.
Uses synthetic WorldState objects, so no camera, YOLO model, or robot is needed.
"""

import unittest

from controller.nearest import find_nearest, create_nodes_and_edges
from test.world_state_helpers import world_state


class NearestUnitTests(unittest.TestCase):


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
        # Since the graph now only adds edges from the robot to the targets (skipping goal and itself)
        # There are 3 white balls + 1 orange ball = 4 edges
        self.assertEqual(graph.number_of_edges(), 4)

    def test_find_nearest_returns_nearest_white_ball(self):
        world = world_state(
            robot=(20.0, 20.0),
            robot_px=(100, 100),
            robot_angle=0.0,
            white_balls=[(40.0, 20.0, "open"), (80.0, 20.0, "open")],
            white_balls_px=[(220, 100, "open"), (440, 100, "open")],
            ob=(150.0, 80.0, "open"),
            ob_px=(800, 400, "open"),
        )

        target = find_nearest(world)
        self.assertIsNotNone(target)
        self.assertTrue(target["id"].startswith("wb_"))
        
    def test_find_nearest_returns_orange_when_no_whites(self):
        world = world_state(
            robot=(20.0, 20.0),
            robot_px=(100, 100),
            robot_angle=0.0,
            ob=(150.0, 80.0, "open"),
            ob_px=(800, 400, "open"),
        )

        target = find_nearest(world)
        self.assertIsNotNone(target)
        self.assertEqual(target["id"], "ob")

    def test_find_nearest_returns_none_when_robot_is_missing(self):
        world = world_state(
            white_balls=[(40.0, 20.0, "open")],
            white_balls_px=[(220, 100, "open")],
        )

        self.assertIsNone(find_nearest(world))


if __name__ == "__main__":
    unittest.main()
