"""
test_calibration_tracker_unit.py - offline unit test for EMA calibration tracking.
Verifies that new measurements are blended into the stored ratio.
"""

import unittest
from unittest.mock import mock_open, patch

from controller.calibration_tracker import ALPHA, CalibrationTracker, save_calibration_to_config


class CalibrationTrackerUnitTests(unittest.TestCase):
    def test_update_blends_new_measurement_with_existing_ratio(self):
        tracker = CalibrationTracker(100.0)

        updated = tracker.update(200.0)

        self.assertAlmostEqual(updated, ALPHA * 200.0 + (1 - ALPHA) * 100.0)
        self.assertAlmostEqual(tracker.ratio, updated)

    def test_save_calibration_to_config_rewrites_only_calibration_values(self):
        config_text = (
            "PIXELS_PER_ROTATION = 1.0\n"
            "DEGREES_PER_ROTATION_LEFT = 2.0\n"
            "DEGREES_PER_ROTATION_RIGHT = 3.0\n"
            "OTHER_VALUE = 4.0\n"
        )
        opened = mock_open(read_data=config_text)

        with patch("builtins.open", opened), \
             patch("controller.calibration_tracker.calibration_pixels.ratio", 10.123), \
             patch("controller.calibration_tracker.calibration_angle_left.ratio", 20.456), \
             patch("controller.calibration_tracker.calibration_angle_right.ratio", 30.789):
            values = save_calibration_to_config()

        written = opened().write.call_args.args[0]
        self.assertEqual(values, (10.123, 20.456, 30.789))
        self.assertIn("PIXELS_PER_ROTATION = 10.12", written)
        self.assertIn("DEGREES_PER_ROTATION_LEFT = 20.46", written)
        self.assertIn("DEGREES_PER_ROTATION_RIGHT = 30.79", written)
        self.assertIn("OTHER_VALUE = 4.0", written)


if __name__ == "__main__":
    unittest.main()

