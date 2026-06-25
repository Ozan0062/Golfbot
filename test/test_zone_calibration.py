"""
test_zone_calibration.py — tests for the zone-based calibration system.
"""

import json
import math
import os
import sys
import tempfile

import pytest

# Ensure project root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.zone_calibration_tracker import (
    get_zone, ZoneCalibrationTracker, _ZoneData, NUM_ZONES, ALPHA,
)


# --------------------------------------------------------------------------
# get_zone
# --------------------------------------------------------------------------

class TestGetZone:
    CENTER = (450, 300)

    def test_top_left(self):
        assert get_zone((100, 100), self.CENTER) == 0

    def test_top_right(self):
        assert get_zone((600, 100), self.CENTER) == 1

    def test_bottom_left(self):
        assert get_zone((100, 400), self.CENTER) == 2

    def test_bottom_right(self):
        assert get_zone((600, 400), self.CENTER) == 3

    def test_on_center_x_boundary_goes_right(self):
        assert get_zone((450, 100), self.CENTER) == 1

    def test_on_center_y_boundary_goes_bottom(self):
        assert get_zone((100, 300), self.CENTER) == 2

    def test_on_exact_center_goes_bottom_right(self):
        assert get_zone((450, 300), self.CENTER) == 3

    def test_none_position(self):
        assert get_zone(None, self.CENTER) is None


# --------------------------------------------------------------------------
# _ZoneData
# --------------------------------------------------------------------------

class TestZoneData:
    def test_to_dict_round_trip(self):
        zd = _ZoneData(64.0, 31.0, 29.0)
        zd.samples_drive = 5
        zd.samples_turn_left = 3
        zd.samples_turn_right = 2
        d = zd.to_dict()
        restored = _ZoneData.from_dict(d, 99, 99, 99)
        assert restored.px_per_rot == pytest.approx(64.0)
        assert restored.deg_per_rot_left == pytest.approx(31.0)
        assert restored.deg_per_rot_right == pytest.approx(29.0)
        assert restored.samples_drive == 5

    def test_from_dict_uses_fallback(self):
        restored = _ZoneData.from_dict({}, 64.0, 31.0, 29.0)
        assert restored.px_per_rot == pytest.approx(64.0)
        assert restored.samples_drive == 0


# --------------------------------------------------------------------------
# ZoneCalibrationTracker
# --------------------------------------------------------------------------

class TestZoneCalibrationTracker:

    def _make_tracker(self, path=None):
        if path is None:
            path = os.path.join(tempfile.mkdtemp(), "test_zone_cal.json")
        return ZoneCalibrationTracker(
            center_px=(450, 300),
            fallback_px=64.0,
            fallback_left=31.0,
            fallback_right=29.0,
            path=path,
        )

    # -- EMA update tests --

    def test_update_drive_applies_ema(self):
        t = self._make_tracker()
        t.update_drive(0, 70.0)
        expected = ALPHA * 70.0 + (1 - ALPHA) * 64.0
        assert t.zones[0].px_per_rot == pytest.approx(expected)
        assert t.zones[0].samples_drive == 1
        # Other zones untouched
        assert t.zones[1].px_per_rot == pytest.approx(64.0)
        assert t.zones[1].samples_drive == 0

    def test_update_turn_left(self):
        t = self._make_tracker()
        t.update_turn(1, 35.0, "LEFT")
        expected = ALPHA * 35.0 + (1 - ALPHA) * 31.0
        assert t.zones[1].deg_per_rot_left == pytest.approx(expected)
        assert t.zones[1].samples_turn_left == 1
        # Right unchanged
        assert t.zones[1].deg_per_rot_right == pytest.approx(29.0)

    def test_update_turn_right(self):
        t = self._make_tracker()
        t.update_turn(2, 33.0, "RIGHT")
        expected = ALPHA * 33.0 + (1 - ALPHA) * 29.0
        assert t.zones[2].deg_per_rot_right == pytest.approx(expected)
        assert t.zones[2].samples_turn_right == 1

    def test_update_none_zone_is_noop(self):
        t = self._make_tracker()
        t.update_drive(None, 70.0)
        for z in t.zones.values():
            assert z.samples_drive == 0

    def test_update_invalid_zone_is_noop(self):
        t = self._make_tracker()
        t.update_drive(99, 70.0)
        for z in t.zones.values():
            assert z.samples_drive == 0

    # -- Lookup tests --

    def test_get_px_per_rotation_with_samples(self):
        t = self._make_tracker()
        t.update_drive(0, 70.0)
        result = t.get_px_per_rotation((100, 100))  # zone 0
        expected = ALPHA * 70.0 + (1 - ALPHA) * 64.0
        assert result == pytest.approx(expected)

    def test_get_px_per_rotation_no_samples_uses_fallback(self):
        t = self._make_tracker()
        result = t.get_px_per_rotation((100, 100))  # zone 0, no samples
        assert result == pytest.approx(64.0)

    def test_get_deg_per_rotation_with_samples(self):
        t = self._make_tracker()
        t.update_turn(3, 33.0, "RIGHT")
        result = t.get_deg_per_rotation((600, 400), "RIGHT")  # zone 3
        expected = ALPHA * 33.0 + (1 - ALPHA) * 29.0
        assert result == pytest.approx(expected)

    def test_get_deg_per_rotation_no_samples_uses_fallback(self):
        t = self._make_tracker()
        result = t.get_deg_per_rotation((600, 400), "LEFT")  # zone 3, no samples
        assert result == pytest.approx(31.0)

    def test_get_px_per_rotation_none_pos(self):
        t = self._make_tracker()
        result = t.get_px_per_rotation(None)
        assert result == pytest.approx(64.0)

    # -- Persistence tests --

    def test_save_and_load(self):
        path = os.path.join(tempfile.mkdtemp(), "test_cal.json")
        t1 = self._make_tracker(path)
        t1.update_drive(0, 70.0)
        t1.update_drive(0, 72.0)
        t1.update_turn(2, 35.0, "LEFT")
        t1.save()

        # Verify file exists and is valid JSON
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "zones" in data
        assert "0" in data["zones"]

        # Load into a fresh tracker
        t2 = self._make_tracker(path)
        t2.load()
        assert t2.zones[0].px_per_rot == pytest.approx(t1.zones[0].px_per_rot)
        assert t2.zones[0].samples_drive == 2
        assert t2.zones[2].deg_per_rot_left == pytest.approx(t1.zones[2].deg_per_rot_left)
        assert t2.zones[2].samples_turn_left == 1
        # Zones that were not touched should be at fallback
        assert t2.zones[3].px_per_rot == pytest.approx(64.0)

    def test_load_missing_file_uses_defaults(self):
        path = os.path.join(tempfile.mkdtemp(), "nonexistent.json")
        t = self._make_tracker(path)
        t.load()  # should not raise
        assert t.zones[0].px_per_rot == pytest.approx(64.0)

    def test_load_corrupt_file_uses_defaults(self):
        path = os.path.join(tempfile.mkdtemp(), "corrupt.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        t = self._make_tracker(path)
        t.load()  # should not raise
        assert t.zones[0].px_per_rot == pytest.approx(64.0)

    # -- Multiple EMA updates converge --

    def test_many_updates_converge(self):
        t = self._make_tracker()
        for _ in range(50):
            t.update_drive(0, 70.0)
        # After many updates, should be very close to the measurement (70.0)
        assert t.zones[0].px_per_rot == pytest.approx(70.0, abs=0.1)

    # -- Summary table --

    def test_summary_table_is_string(self):
        t = self._make_tracker()
        table = t.summary_table()
        assert isinstance(table, str)
        assert "Zone" in table
        assert "TL" in table
        assert "BR" in table
