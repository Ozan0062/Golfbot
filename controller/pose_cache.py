"""
pose_cache.py — ArUco pose with timeout fallback.

The camera can miss the ArUco marker for a frame or two.  PoseCache keeps the
last valid reading and returns it until it goes stale (> POSE_TIMEOUT_S old).
After that it returns None so the state machine knows to wait.
"""

import time
from dataclasses import dataclass
from typing import Optional


POSE_TIMEOUT_S = 0.5


@dataclass
class Pose:
    pos:   tuple   # (x, y) in cm — used for angle and TSP maths
    px:    tuple   # (x, y) in pixels — used for drive distance
    angle: float   # heading in degrees


class PoseCache:

    def __init__(self):
        self._pose:       Optional[Pose] = None
        self._last_seen:  float          = 0.0

    def update(self, world: dict) -> Optional[Pose]:
        """
        Feed a new world dict.  Returns the best available Pose, or None if
        the marker hasn't been seen recently enough to trust.
        """
        if world.get("robot") is not None and world.get("robot_angle") is not None:
            self._pose = Pose(
                pos=world["robot"],
                px=world.get("robot_px"),
                angle=world["robot_angle"],
            )
            self._last_seen = time.time()

        if self._pose is None:
            return None

        age = time.time() - self._last_seen
        return self._pose if age < POSE_TIMEOUT_S else None
