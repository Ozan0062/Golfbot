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


# DECISION: 0.5s timeout — if the marker hasn't been seen for half a second,
# the cached pose is too stale to trust (robot may have drifted).
POSE_TIMEOUT_S = 0.5

# DECISION: 0.2s settle window after each move. The EV3 reports "done"
# before the robot fully stops coasting, so ArUco readings taken immediately
# after a move show the robot mid-slide with a wrong angle. This blackout
# window ignores all detections until the robot has physically settled.
SETTLE_S = 0.2


@dataclass
class Pose:
    pos:   tuple   # (x, y) in cm — used for angle and TSP maths
    px:    tuple   # (x, y) in pixels — used for drive distance
    angle: float   # heading in degrees


class PoseCache:

    def __init__(self):
        self._pose:        Optional[Pose] = None
        self._last_seen:   float          = 0.0
        self._valid_after: float          = 0.0   # ignore detections before this time

    def invalidate(self):
        """
        Call after any blocking robot move.  Clears the cached pose and blocks
        new detections for SETTLE_S so the robot can fully stop before the next
        ArUco reading is trusted.
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
            remaining = self._valid_after - now
            print(f"[POSE] Settling — ignoring detection, {remaining*1000:.0f}ms left")
            return None

        # First frame after settle window expires
        if self._pose is None and self._valid_after > 0:
            print(f"[POSE] Settle complete — waiting for ArUco")

        if world.get("robot") is not None and world.get("robot_angle") is not None:
            was_empty = self._pose is None
            self._pose      = Pose(
                pos=world["robot"],
                px=world.get("robot_px"),
                angle=world["robot_angle"],
            )
            self._last_seen = now
            if was_empty:
                print(f"[POSE] Fresh lock — angle={self._pose.angle:.1f}°  pos={self._pose.pos}")

        if self._pose is None:
            return None

        age = now - self._last_seen
        return self._pose if age < POSE_TIMEOUT_S else None
