#!/usr/bin/env python3
"""
Tests for robot/mov_control.py

ev3dev2 is not available outside the brick, so we stub the whole package
before importing mov_control.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Stub ev3dev2 before any import of mov_control
# ---------------------------------------------------------------------------

def _build_ev3dev2_stub():
    """Return a minimal sys.modules stub for ev3dev2."""
    ev3dev2       = types.ModuleType("ev3dev2")
    ev3dev2_motor = types.ModuleType("ev3dev2.motor")

    mock_tank = MagicMock()
    MoveTank  = MagicMock(return_value=mock_tank)

    class SpeedPercent:                      # preserve the sign arithmetic
        def __init__(self, pct): self.pct = pct
        def __repr__(self): return f"SpeedPercent({self.pct})"
        def __eq__(self, other):
            return isinstance(other, SpeedPercent) and self.pct == other.pct

    ev3dev2_motor.MoveTank     = MoveTank
    ev3dev2_motor.SpeedPercent = SpeedPercent
    ev3dev2_motor.OUTPUT_A     = "A"
    ev3dev2_motor.OUTPUT_D     = "D"

    sys.modules["ev3dev2"]       = ev3dev2
    sys.modules["ev3dev2.motor"] = ev3dev2_motor

    return mock_tank, SpeedPercent


_mock_tank, SpeedPercent = _build_ev3dev2_stub()

# Now safe to import
import importlib
import robot.mov_control as mov     # noqa: E402  (import after stub)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dps(pct):
    return (pct / 100) * mov.LARGE_MOTOR_MAX_DPS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDriveForward(unittest.TestCase):

    def setUp(self):
        _mock_tank.reset_mock()

    def test_calls_on_for_rotations(self):
        mov.drive_forward(2.0)
        _mock_tank.on_for_rotations.assert_called_once()

    def test_correct_rotations(self):
        mov.drive_forward(3.5)
        _, _, rotations = _mock_tank.on_for_rotations.call_args[0]
        self.assertEqual(rotations, 3.5)

    def test_both_wheels_same_direction(self):
        mov.drive_forward(1.0)
        left, right, _ = _mock_tank.on_for_rotations.call_args[0]
        # signs must match (both forward)
        self.assertEqual(left.pct > 0, right.pct > 0)

    def test_zero_rotations_does_not_raise(self):
        try:
            mov.drive_forward(0)
        except Exception as e:
            self.fail(f"drive_forward(0) raised {e}")


class TestDriveBackward(unittest.TestCase):

    def setUp(self):
        _mock_tank.reset_mock()

    def test_calls_on_for_rotations(self):
        mov.drive_backward(2.0)
        _mock_tank.on_for_rotations.assert_called_once()

    def test_correct_rotations(self):
        mov.drive_backward(1.5)
        _, _, rotations = _mock_tank.on_for_rotations.call_args[0]
        self.assertEqual(rotations, 1.5)

    def test_direction_opposite_to_forward(self):
        mov.drive_forward(1.0)
        fwd_left, _, _ = _mock_tank.on_for_rotations.call_args[0]
        _mock_tank.reset_mock()

        mov.drive_backward(1.0)
        bwd_left, _, _ = _mock_tank.on_for_rotations.call_args[0]

        self.assertNotEqual(fwd_left.pct > 0, bwd_left.pct > 0)


class TestTurnLeft(unittest.TestCase):

    def setUp(self):
        _mock_tank.reset_mock()

    def test_calls_on_for_rotations(self):
        mov.turn_left(1.0)
        _mock_tank.on_for_rotations.assert_called_once()

    def test_correct_rotations(self):
        mov.turn_left(0.5)
        _, _, rotations = _mock_tank.on_for_rotations.call_args[0]
        self.assertEqual(rotations, 0.5)

    def test_wheels_opposite_directions(self):
        mov.turn_left(1.0)
        left, right, _ = _mock_tank.on_for_rotations.call_args[0]
        # one wheel forward, one back
        self.assertNotEqual(left.pct > 0, right.pct > 0)

    def test_left_wheel_backward_right_wheel_forward(self):
        mov.turn_left(1.0)
        left, right, _ = _mock_tank.on_for_rotations.call_args[0]
        self.assertLess(left.pct, 0)
        self.assertGreater(right.pct, 0)


class TestTurnRight(unittest.TestCase):

    def setUp(self):
        _mock_tank.reset_mock()

    def test_calls_on_for_rotations(self):
        mov.turn_right(1.0)
        _mock_tank.on_for_rotations.assert_called_once()

    def test_correct_rotations(self):
        mov.turn_right(0.5)
        _, _, rotations = _mock_tank.on_for_rotations.call_args[0]
        self.assertEqual(rotations, 0.5)

    def test_wheels_opposite_directions(self):
        mov.turn_right(1.0)
        left, right, _ = _mock_tank.on_for_rotations.call_args[0]
        self.assertNotEqual(left.pct > 0, right.pct > 0)

    def test_left_wheel_forward_right_wheel_backward(self):
        mov.turn_right(1.0)
        left, right, _ = _mock_tank.on_for_rotations.call_args[0]
        self.assertGreater(left.pct, 0)
        self.assertLess(right.pct, 0)

    def test_opposite_to_turn_left(self):
        mov.turn_left(1.0)
        left_l, right_l, _ = _mock_tank.on_for_rotations.call_args[0]
        _mock_tank.reset_mock()

        mov.turn_right(1.0)
        left_r, right_r, _ = _mock_tank.on_for_rotations.call_args[0]

        self.assertEqual(left_l.pct, -left_r.pct)
        self.assertEqual(right_l.pct, -right_r.pct)


class TestStop(unittest.TestCase):

    def setUp(self):
        _mock_tank.reset_mock()

    def test_calls_off_with_brake(self):
        mov.stop()
        _mock_tank.off.assert_called_once_with(brake=True)


if __name__ == "__main__":
    unittest.main()
