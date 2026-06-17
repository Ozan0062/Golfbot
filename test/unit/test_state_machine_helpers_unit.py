"""
test_state_machine_helpers_unit.py - unit tests for state-machine helper functions.
Verifies distance and rotation conversion helpers without controller state.
"""

import unittest

from controller.state_machine import _angle_to_rotations, _distance_px, _px_to_rotations


class StateMachineHelperTests(unittest.TestCase):
    def test_module_helpers_handle_none_and_rotation_conversion(self):
        self.assertEqual(_distance_px(None, (1, 2)), 0.0)
        self.assertEqual(_distance_px((1, 2), None), 0.0)
        self.assertAlmostEqual(_distance_px((0, 0), (3, 4)), 5.0)
        self.assertGreater(_px_to_rotations(100), 0)
        self.assertGreater(_angle_to_rotations(25), 0)
        self.assertGreater(_angle_to_rotations(-25), 0)


if __name__ == "__main__":
    unittest.main()

