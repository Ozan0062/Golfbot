"""
test_calibration_tracker_unit.py - offline unit test for EMA calibration tracking.
Verifies that new measurements are blended into the stored ratio.
"""

import unittest

from controller.calibration_tracker import ALPHA, CalibrationTracker


class CalibrationTrackerUnitTests(unittest.TestCase):
    def test_update_blends_new_measurement_with_existing_ratio(self):
        tracker = CalibrationTracker(100.0)

        updated = tracker.update(200.0)

        self.assertAlmostEqual(updated, ALPHA * 200.0 + (1 - ALPHA) * 100.0)
        self.assertAlmostEqual(tracker.ratio, updated)


if __name__ == "__main__":
    unittest.main()

