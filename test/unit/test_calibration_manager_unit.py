"""
test_calibration_manager_unit.py - offline unit tests for runtime calibration updates.
Mocks shared calibration trackers so no robot movement or camera frame is needed.
"""

import unittest
from unittest.mock import patch

from controller.calibration_manager import CalibrationManager


class FakeTracker:
    def __init__(self):
        self.measurements = []
        self.ratio = 0.0

    def update(self, measurement):
        self.measurements.append(measurement)
        self.ratio = measurement


class CalibrationManagerUnitTests(unittest.TestCase):
    def test_drive_measurement_updates_pixel_tracker_when_rotation_is_large_enough(self):
        pixels = FakeTracker()
        manager = CalibrationManager()
        manager.record_drive((0, 0), 2.0)

        with patch("builtins.print"), patch(
            "controller.calibration_manager.calibration_pixels", pixels
        ):
            manager.consume((6, 8), 0.0)

        self.assertEqual(pixels.measurements, [5.0])

    def test_small_drive_measurement_is_ignored(self):
        pixels = FakeTracker()
        manager = CalibrationManager()
        manager.record_drive((0, 0), 0.1)

        with patch("controller.calibration_manager.calibration_pixels", pixels):
            manager.consume((6, 8), 0.0)

        self.assertEqual(pixels.measurements, [])

    def test_left_and_right_turn_measurements_update_their_own_trackers(self):
        left = FakeTracker()
        right = FakeTracker()
        manager = CalibrationManager()

        with patch("builtins.print"), patch(
            "controller.calibration_manager.calibration_angle_left", left
        ), patch("controller.calibration_manager.calibration_angle_right", right):
            manager.record_turn(350.0, 2.0, "LEFT")
            manager.consume((0, 0), 10.0)
            manager.record_turn(10.0, 4.0, "RIGHT")
            manager.consume((0, 0), 50.0)

        self.assertEqual(left.measurements, [10.0])
        self.assertEqual(right.measurements, [10.0])


if __name__ == "__main__":
    unittest.main()

