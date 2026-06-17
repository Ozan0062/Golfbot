"""
test_calibration_unit.py - offline unit tests for drive and turn calibration helpers.
Verifies distance/angle conversions without moving the robot.
"""

import unittest

from controller.drive_calibration import (
    measure_cm_per_rotation,
    measure_degrees_per_rotation,
    measure_pixels_per_rotation,
    rotations_for_angle,
    rotations_for_distance,
)


class CalibrationUnitTests(unittest.TestCase):
    def test_calibration_helpers_convert_distance_and_wrapped_heading_to_rotations(self):
        self.assertAlmostEqual(
            measure_pixels_per_rotation((0, 0), (6, 8), rotations=2.0),
            5.0,
        )
        self.assertAlmostEqual(
            measure_cm_per_rotation((0, 0), (3, 4), rotations=2.0),
            2.5,
        )
        self.assertAlmostEqual(
            rotations_for_distance(distance_px=235.0, pixels_per_rotation=47.0),
            5.0,
        )
        self.assertAlmostEqual(
            rotations_for_angle(angle_deg=-50.0, degrees_per_rotation=25.0),
            2.0,
        )
        self.assertAlmostEqual(
            measure_degrees_per_rotation(
                start_heading=350.0,
                end_heading=10.0,
                rotations=2.0,
            ),
            10.0,
        )

    def test_calibration_helpers_raise_when_dividing_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            rotations_for_distance(distance_px=235.0, pixels_per_rotation=0.0)

        with self.assertRaises(ZeroDivisionError):
            measure_degrees_per_rotation(
                start_heading=350.0,
                end_heading=10.0,
                rotations=0.0,
            )


if __name__ == "__main__":
    unittest.main()
