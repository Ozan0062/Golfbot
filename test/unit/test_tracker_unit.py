"""
test_tracker_unit.py - offline unit tests for pixel-to-cm conversion and object extraction.
Uses mock detections instead of YOLO, camera, or robot hardware.
"""

import unittest

from vision.tracker import (
    correct_robot_height,
    extract_objects,
    filter_detections_near_robot,
    pixels_to_cm,
    robot_px_to_cm,
)
from vision.detector import Node_object
from config import FIELD_WIDTH_CM, FIELD_HEIGHT_CM


class TrackerUnitTests(unittest.TestCase):
    def assertTupleAlmostEqual(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_pixels_to_cm_places_center_of_warped_field_at_center_of_real_field(self):
        detections = [
            Node_object(class_name="wb", center=(320, 240), size=(0,0), confidence=0.9),
        ]

        converted = pixels_to_cm(detections, image_width=640, image_height=480)

        self.assertTupleAlmostEqual(converted[0].position_cm, (FIELD_WIDTH_CM / 2, FIELD_HEIGHT_CM / 2))

    def test_robot_px_to_cm_returns_none_when_aruco_is_missing(self):
        self.assertIsNone(robot_px_to_cm(None, image_width=640, image_height=480))
        self.assertTupleAlmostEqual(
            robot_px_to_cm((320, 240), image_width=640, image_height=480),
            (FIELD_WIDTH_CM / 2, FIELD_HEIGHT_CM / 2),
        )

    def test_extract_objects_builds_world_objects_from_mock_yolo_detections(self):
        detections = [
            Node_object(class_name="wb", center=(100, 100), size=(0,0), confidence=0.8),
            Node_object(class_name="ob", center=(200, 100), size=(0,0), confidence=0.9),
            Node_object(class_name="cross", center=(300, 200), size=(0,0), confidence=0.95),
        ]

        objects = extract_objects(pixels_to_cm(detections, image_width=640, image_height=480))

        self.assertTupleAlmostEqual(objects.white_balls[0][:2], (26.5625, 25.9375))
        self.assertEqual(objects.white_balls[0][2], "open")
        self.assertEqual(objects.white_balls_px, [(100, 100, "open")])
        self.assertTupleAlmostEqual(objects.ob[:2], (53.125, 25.9375))
        self.assertEqual(objects.ob[2], "open")
        self.assertEqual(objects.ob_px, (200, 100, "open"))
        self.assertTupleAlmostEqual(objects.cross, (79.6875, 51.875))
        self.assertEqual(objects.cross_px, (300, 200))

    def test_extract_objects_ignores_unknown_classes(self):
        detections = [
            Node_object(class_name="unknown", center=(123, 456), size=(0,0), confidence=0.99),
        ]

        objects = extract_objects(pixels_to_cm(detections, image_width=640, image_height=480))

        self.assertEqual(objects.white_balls, [])
        self.assertIsNone(objects.ob)
        self.assertIsNone(objects.cross)

    def test_extract_objects_keeps_highest_confidence_orange_ball_and_cross(self):
        detections = [
            Node_object(class_name="ob", center=(100, 100), size=(0,0), confidence=0.4),
            Node_object(class_name="ob", center=(200, 200), size=(0,0), confidence=0.9),
            Node_object(class_name="cross", center=(300, 300), size=(0,0), confidence=0.3),
            Node_object(class_name="cross", center=(400, 400), size=(0,0), confidence=0.8),
        ]

        objects = extract_objects(pixels_to_cm(detections, image_width=640, image_height=480))

        self.assertTupleAlmostEqual(objects.ob[:2], (53.125, 51.875))
        self.assertEqual(objects.ob[2], "open")
        self.assertEqual(objects.ob_px, (200, 200, "open"))
        self.assertTupleAlmostEqual(objects.cross, (106.25, 103.75))
        self.assertEqual(objects.cross_px, (400, 400))

    def test_filter_detections_near_robot_removes_near_ball_but_keeps_cross(self):
        near_ball = Node_object(class_name="wb", center=(105, 105), size=(0, 0), confidence=0.9)
        far_ball = Node_object(class_name="ob", center=(200, 200), size=(0, 0), confidence=0.9)
        cross = Node_object(class_name="cross", center=(102, 102), size=(0, 0), confidence=0.9)

        filtered = filter_detections_near_robot([near_ball, far_ball, cross], (100, 100), radius=20)

        self.assertNotIn(near_ball, filtered)
        self.assertIn(far_ball, filtered)
        self.assertIn(cross, filtered)
        self.assertGreater(far_ball.dist_from_robot, 20)

    def test_filter_detections_returns_original_list_when_robot_is_missing(self):
        detections = [Node_object(class_name="wb", center=(105, 105), size=(0, 0), confidence=0.9)]

        self.assertIs(filter_detections_near_robot(detections, None), detections)

    def test_correct_robot_height_keeps_invalid_or_centered_inputs_stable(self):
        self.assertIsNone(
            correct_robot_height(None, (100, 100), 178.0, 19.8, 900, 600, 169.0, 124.5)
        )
        self.assertEqual(
            correct_robot_height((100, 100), (100, 100), 178.0, 19.8, 900, 600, 169.0, 124.5),
            (100, 100),
        )


if __name__ == "__main__":
    unittest.main()
