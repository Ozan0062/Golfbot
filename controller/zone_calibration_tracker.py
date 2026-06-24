"""
zone_calibration_tracker.py — per-zone EMA calibration for the GolfBot.

The field is divided into 4 quadrants around the field centre (ZONE_CENTER_PX).
Each zone maintains its own px-per-rotation and deg-per-rotation (left / right)
estimates, refined with EMA (alpha = 0.15) every time a drive or turn is
measured while the robot stays inside a single zone.

On import the module loads zone_calibration.json (if it exists) so a warm start
is free.  Call save() to persist the current state.

Zone numbering:

    ┌────────┬────────┐
    │ Zone 0 │ Zone 1 │
    │  (TL)  │  (TR)  │
    ├────────┼────────┤
    │ Zone 2 │ Zone 3 │
    │  (BL)  │  (BR)  │
    └────────┴────────┘
"""

import json
import os
from datetime import datetime

from golfbot_logger import get_logger

log = get_logger(__name__)

ALPHA = 0.15              # EMA smoothing factor (same as global tracker)
NUM_ZONES = 4
_ZONE_NAMES = {0: "TL", 1: "TR", 2: "BL", 3: "BR"}


# ---------------------------------------------------------------------------
# Zone lookup
# ---------------------------------------------------------------------------

def get_zone(pos_px, center_px):
    """
    Return zone index 0-3 for a warped-pixel position.

    0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.
    Returns None if pos_px is None.
    """
    if pos_px is None:
        return None
    col = 0 if pos_px[0] < center_px[0] else 1
    row = 0 if pos_px[1] < center_px[1] else 1
    return row * 2 + col


# ---------------------------------------------------------------------------
# Per-zone data holder
# ---------------------------------------------------------------------------

class _ZoneData:
    """Holds calibration ratios and sample counts for one zone."""

    __slots__ = (
        "px_per_rot", "deg_per_rot_left", "deg_per_rot_right",
        "samples_drive", "samples_turn_left", "samples_turn_right",
    )

    def __init__(self, px_per_rot, deg_per_rot_left, deg_per_rot_right):
        self.px_per_rot       = float(px_per_rot)
        self.deg_per_rot_left  = float(deg_per_rot_left)
        self.deg_per_rot_right = float(deg_per_rot_right)
        self.samples_drive      = 0
        self.samples_turn_left  = 0
        self.samples_turn_right = 0

    def to_dict(self):
        return {
            "px_per_rot":        round(self.px_per_rot, 4),
            "deg_per_rot_left":  round(self.deg_per_rot_left, 4),
            "deg_per_rot_right": round(self.deg_per_rot_right, 4),
            "samples_drive":     self.samples_drive,
            "samples_turn_left": self.samples_turn_left,
            "samples_turn_right": self.samples_turn_right,
        }

    @classmethod
    def from_dict(cls, d, fallback_px, fallback_left, fallback_right):
        z = cls(
            d.get("px_per_rot",        fallback_px),
            d.get("deg_per_rot_left",  fallback_left),
            d.get("deg_per_rot_right", fallback_right),
        )
        z.samples_drive      = d.get("samples_drive", 0)
        z.samples_turn_left  = d.get("samples_turn_left", 0)
        z.samples_turn_right = d.get("samples_turn_right", 0)
        return z


# ---------------------------------------------------------------------------
# Zone calibration tracker
# ---------------------------------------------------------------------------

class ZoneCalibrationTracker:
    """
    Manages per-zone calibration data.

    * ``update_drive(zone, measured)``  — refine px/rot for a zone.
    * ``update_turn(zone, measured, direction)``  — refine deg/rot for a zone.
    * ``get_px_per_rotation(pos_px)``  — zone-aware px/rot lookup.
    * ``get_deg_per_rotation(pos_px, direction)``  — zone-aware deg/rot lookup.
    * ``save()`` / ``load()`` — JSON round-trip.

    When a zone has zero samples it returns the global config fallback.
    """

    def __init__(self, center_px, fallback_px, fallback_left, fallback_right, path):
        self.center_px = tuple(center_px)
        self._fallback_px    = float(fallback_px)
        self._fallback_left  = float(fallback_left)
        self._fallback_right = float(fallback_right)
        self._path = path

        self.zones = {
            i: _ZoneData(fallback_px, fallback_left, fallback_right)
            for i in range(NUM_ZONES)
        }

    # -- EMA updates ----------------------------------------------------------

    def update_drive(self, zone, measured):
        """Update px_per_rotation for *zone* with a new measurement."""
        if zone is None or zone not in self.zones:
            return
        zd = self.zones[zone]
        zd.px_per_rot = ALPHA * measured + (1 - ALPHA) * zd.px_per_rot
        zd.samples_drive += 1
        log.debug("zone %d (%s) drive → %.2f px/rot  (n=%d)",
                  zone, _ZONE_NAMES[zone], zd.px_per_rot, zd.samples_drive)

    def update_turn(self, zone, measured, direction):
        """Update deg_per_rotation for *zone* with a new measurement.

        direction: 'LEFT' or 'RIGHT'.
        """
        if zone is None or zone not in self.zones:
            return
        zd = self.zones[zone]
        if direction == "LEFT":
            zd.deg_per_rot_left = ALPHA * measured + (1 - ALPHA) * zd.deg_per_rot_left
            zd.samples_turn_left += 1
            log.debug("zone %d (%s) turn L → %.2f deg/rot  (n=%d)",
                      zone, _ZONE_NAMES[zone], zd.deg_per_rot_left, zd.samples_turn_left)
        else:
            zd.deg_per_rot_right = ALPHA * measured + (1 - ALPHA) * zd.deg_per_rot_right
            zd.samples_turn_right += 1
            log.debug("zone %d (%s) turn R → %.2f deg/rot  (n=%d)",
                      zone, _ZONE_NAMES[zone], zd.deg_per_rot_right, zd.samples_turn_right)

    # -- Lookups (with fallback) ----------------------------------------------

    def get_px_per_rotation(self, pos_px):
        """Return the zone-specific px/rot for the given pixel position."""
        zone = get_zone(pos_px, self.center_px)
        if zone is not None and self.zones[zone].samples_drive > 0:
            return self.zones[zone].px_per_rot
        return self._fallback_px

    def get_deg_per_rotation(self, pos_px, direction):
        """Return the zone-specific deg/rot for the given pixel position and turn direction."""
        zone = get_zone(pos_px, self.center_px)
        if zone is not None:
            zd = self.zones[zone]
            if direction == "LEFT" and zd.samples_turn_left > 0:
                return zd.deg_per_rot_left
            if direction == "RIGHT" and zd.samples_turn_right > 0:
                return zd.deg_per_rot_right
        return self._fallback_left if direction == "LEFT" else self._fallback_right

    # -- Persistence ----------------------------------------------------------

    def save(self):
        """Write current state to JSON."""
        data = {
            "center_px": list(self.center_px),
            "zones": {str(i): zd.to_dict() for i, zd in self.zones.items()},
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Zone calibration saved to %s", self._path)

    def load(self):
        """Load state from JSON (if the file exists)."""
        if not os.path.exists(self._path):
            log.info("No zone calibration file found at %s — using defaults", self._path)
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_center = data.get("center_px")
            if saved_center is not None:
                self.center_px = tuple(saved_center)
            for key, zd_dict in data.get("zones", {}).items():
                idx = int(key)
                if idx in self.zones:
                    self.zones[idx] = _ZoneData.from_dict(
                        zd_dict, self._fallback_px,
                        self._fallback_left, self._fallback_right,
                    )
            total = sum(zd.samples_drive for zd in self.zones.values())
            log.info("Zone calibration loaded from %s  (%d total drive samples)", self._path, total)
        except Exception as exc:
            log.warning("Failed to load zone calibration from %s: %s — using defaults", self._path, exc)

    # -- Pretty-print ---------------------------------------------------------

    def summary_table(self):
        """Return a multi-line summary string suitable for console output."""
        lines = [
            "Zone  │ px/rot  │ L deg/rot │ R deg/rot │ n_drv │ n_L  │ n_R",
            "──────┼─────────┼───────────┼───────────┼───────┼──────┼──────",
        ]
        for i in range(NUM_ZONES):
            zd = self.zones[i]
            lines.append(
                f"  {i} {_ZONE_NAMES[i]} │ {zd.px_per_rot:7.2f} │ {zd.deg_per_rot_left:9.2f} │ "
                f"{zd.deg_per_rot_right:9.2f} │ {zd.samples_drive:5d} │ {zd.samples_turn_left:4d} │ {zd.samples_turn_right:4d}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton — created from config on first import
# ---------------------------------------------------------------------------

def _create_tracker():
    from config import (
        PIXELS_PER_ROTATION, DEGREES_PER_ROTATION_LEFT, DEGREES_PER_ROTATION_RIGHT,
        ZONE_CENTER_PX, ZONE_CALIBRATION_FILE,
    )
    tracker = ZoneCalibrationTracker(
        center_px=ZONE_CENTER_PX,
        fallback_px=PIXELS_PER_ROTATION,
        fallback_left=DEGREES_PER_ROTATION_LEFT,
        fallback_right=DEGREES_PER_ROTATION_RIGHT,
        path=ZONE_CALIBRATION_FILE,
    )
    tracker.load()
    return tracker


zone_tracker = _create_tracker()
