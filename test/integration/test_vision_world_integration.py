"""
test_vision_world_integration.py - integration test for detections -> world dict.
Uses fake YOLO detections to verify navigation inputs without camera/model hardware.
"""

import unittest

from vision.tracker import build_world_state, extract_objects, pixels_to_cm
from config import FIELD_WIDTH_CM, FIELD_HEIGHT_CM, WARPED_WIDTH, WARPED_HEIGHT


class VisionWorldIntegrationTests(unittest.TestCase):
    def assertTupleAlmostEqual(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_vision_detections_to_world_state_contains_navigation_inputs(self):
        from vision.detector import Node_object
        detections = [
            Node_object(class_name="wb", center=(WARPED_WIDTH / 2, WARPED_HEIGHT / 2), size=(0,0), confidence=0.9),
            Node_object(class_name="ob", center=(WARPED_WIDTH, WARPED_HEIGHT), size=(0,0), confidence=0.8),
            Node_object(class_name="cross", center=(WARPED_WIDTH / 4, WARPED_HEIGHT / 4), size=(0,0), confidence=0.7),
        ]

        world = extract_objects(pixels_to_cm(detections, WARPED_WIDTH, WARPED_HEIGHT))

        self.assertTupleAlmostEqual(world.white_balls[0][:2], (FIELD_WIDTH_CM / 2, FIELD_HEIGHT_CM / 2))
        self.assertEqual(world.white_balls_px[0][:2], (WARPED_WIDTH / 2, WARPED_HEIGHT / 2))
        self.assertEqual(world.white_balls[0][2], "open")
        self.assertTupleAlmostEqual(world.ob[:2], (FIELD_WIDTH_CM, FIELD_HEIGHT_CM))
        self.assertEqual(world.ob_px[:2], (WARPED_WIDTH, WARPED_HEIGHT))
        self.assertEqual(world.ob[2], "corner")
        self.assertTupleAlmostEqual(world.cross, (FIELD_WIDTH_CM / 4, FIELD_HEIGHT_CM / 4))
        self.assertEqual(world.cross_px, (WARPED_WIDTH / 4, WARPED_HEIGHT / 4))

    def test_build_world_state_defaults_cross_to_field_center_when_detection_is_missing(self):
        from vision.detector import Node_object
        detections = [
            Node_object(class_name="wb", center=(100, 100), size=(0,0), confidence=0.9),
        ]

        world = build_world_state(detections, robot_center=(200, 200), robot_angle=0.0,
                                  image_w=WARPED_WIDTH, image_h=WARPED_HEIGHT)

        self.assertEqual(world.cross_px, (WARPED_WIDTH / 2, WARPED_HEIGHT / 2))
        self.assertTupleAlmostEqual(world.cross, (FIELD_WIDTH_CM / 2, FIELD_HEIGHT_CM / 2))


if __name__ == "__main__":
    unittest.main()
