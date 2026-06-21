"""
test_vision_world_integration.py - integration test for detections -> world dict.
Uses fake YOLO detections to verify navigation inputs without camera/model hardware.
"""

import unittest

from vision.tracker import extract_objects, pixels_to_cm
from config import FIELD_WIDTH_CM, FIELD_HEIGHT_CM


class VisionWorldIntegrationTests(unittest.TestCase):
    def assertTupleAlmostEqual(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_vision_detections_to_world_dict_contains_navigation_inputs(self):
        detections = [
            {"class_name": "wb", "center": (320, 240), "confidence": 0.9},
            {"class_name": "ob", "center": (640, 480), "confidence": 0.8},
            {"class_name": "cross", "center": (160, 120), "confidence": 0.7},
        ]

        world = extract_objects(pixels_to_cm(detections, 640, 480))

        self.assertTupleAlmostEqual(world["white_balls"][0], (FIELD_WIDTH_CM / 2, FIELD_HEIGHT_CM / 2))
        self.assertEqual(world["white_balls_px"], [(320, 240)])
        self.assertTupleAlmostEqual(world["ob"], (FIELD_WIDTH_CM, FIELD_HEIGHT_CM))
        self.assertEqual(world["ob_px"], (640, 480))
        self.assertTupleAlmostEqual(world["cross"], (FIELD_WIDTH_CM / 4, FIELD_HEIGHT_CM / 4))
        self.assertEqual(world["cross_px"], (160, 120))


if __name__ == "__main__":
    unittest.main()
