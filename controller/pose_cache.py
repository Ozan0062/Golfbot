"""
pose_cache.py — ArUco pose with timeout fallback.

The camera can miss the ArUco marker for a frame or two.  PoseCache keeps the
last valid reading and returns it until it goes stale (> POSE_TIMEOUT_S old).
After that it returns None so the state machine knows to wait.

After a blocking move, call invalidate().  This enforces a SETTLE_S blackout
window: all ArUco detections are ignored until the robot has physically stopped
coasting.  Without this, the camera catches mid-coast frames and the angle
reading is wrong.
"""

import time
from dataclasses import dataclass
from typing import Optional

from golfbot_logger import get_logger

log = get_logger(__name__)

POSE_TIMEOUT_S = 0.5 # If the ArUco marker hasn't been seen for this long, consider the pose unknown.
SETTLE_S = 0.2  # After a blocking move, ignore ArUco readings for this long to let the robot settle before trusting the pose again.


@dataclass
class Pose:
    pos:   tuple   # (x, y) in cm — used for angle/bearing maths
    px:    tuple   # (x, y) in pixels — used for drive distance
    angle: float   # heading in degrees


class PoseCache:

    def __init__(self):
        self._pose:        Optional[Pose] = None
        self._last_seen:   float          = 0.0
        self._valid_after: float          = 0.0   # ignore detections before this time

    def invalidate(self):
        """
        Call after a blocking move to enforce a settle window.
        """
        self._pose        = None
        self._last_seen   = 0.0
        self._valid_after = time.time() + SETTLE_S

    def update(self, world: dict) -> Optional[Pose]:
        """
        Feed a new world dict.  Returns the best available Pose, or None if
        the marker hasn't been seen recently enough to trust.
        """
        now = time.time()

        if now < self._valid_after:
            log.debug("Settling — ignoring detection, %.0f ms left", (self._valid_after - now) * 1000)
            return None

        # First frame after settle window expires
        if self._pose is None and self._valid_after > 0:
            log.debug("Settle complete — waiting for ArUco")

        if world.get("robot") is not None and world.get("robot_angle") is not None:
            was_empty = self._pose is None
            self._pose      = Pose(
                pos=world["robot"],
                px=world.get("robot_px"),
                angle=world["robot_angle"],
            )
            self._last_seen = now
            if was_empty:
                log.debug("Fresh lock — angle=%.1f°  pos=%s  px=%s",
                          self._pose.angle, self._pose.pos, self._pose.px)

        if self._pose is None:
            return None

        age = now - self._last_seen
        return self._pose if age < POSE_TIMEOUT_S else None
