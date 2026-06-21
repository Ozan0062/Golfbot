"""
test_tracker_unit.py - offline unit tests for pixel-to-cm conversion and object extraction.
Uses mock detections instead of YOLO, camera, or robot hardware.
"""

import unittest

from vision.tracker import extract_objects, pixels_to_cm, robot_px_to_cm
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
        detections_cm = [
            Node_object(class_name="wb", center=(100, 100), size=(0,0), confidence=0.8, position_cm=(28.125, 25.0)),
            Node_object(class_name="ob", center=(200, 100), size=(0,0), confidence=0.9, position_cm=(56.25, 25.0)),
            Node_object(class_name="cross", center=(300, 200), size=(0,0), confidence=0.95, position_cm=(84.375, 50.0)),
        ]

        objects = extract_objects(detections_cm)

        self.assertEqual(objects["white_balls"], [(28.125, 25.0)])
        self.assertEqual(objects["white_balls_px"], [(100, 100)])
        self.assertEqual(objects["ob"], (56.25, 25.0))
        self.assertEqual(objects["ob_px"], (200, 100))
        self.assertEqual(objects["cross"], (84.375, 50.0))
        self.assertEqual(objects["cross_px"], (300, 200))

    def test_extract_objects_ignores_unknown_classes(self):
        detections_cm = [
            Node_object(class_name="unknown", center=(123, 456), size=(0,0), confidence=0.99, position_cm=(12.3, 45.6)),
        ]

        objects = extract_objects(detections_cm)

        self.assertEqual(objects["white_balls"], [])
        self.assertIsNone(objects["ob"])
        self.assertIsNone(objects["cross"])

    def test_extract_objects_keeps_highest_confidence_orange_ball_and_cross(self):
        detections_cm = [
            Node_object(class_name="ob", center=(100, 100), size=(0,0), confidence=0.4, position_cm=(10.0, 10.0)),
            Node_object(class_name="ob", center=(200, 200), size=(0,0), confidence=0.9, position_cm=(20.0, 20.0)),
            Node_object(class_name="cross", center=(300, 300), size=(0,0), confidence=0.3, position_cm=(30.0, 30.0)),
            Node_object(class_name="cross", center=(400, 400), size=(0,0), confidence=0.8, position_cm=(40.0, 40.0)),
        ]

        objects = extract_objects(detections_cm)

        self.assertEqual(objects["ob"], (20.0, 20.0))
        self.assertEqual(objects["ob_px"], (200, 200))
        self.assertEqual(objects["cross"], (40.0, 40.0))
        self.assertEqual(objects["cross_px"], (400, 400))


if __name__ == "__main__":
    unittest.main()
