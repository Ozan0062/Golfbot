"""
route_manager.py — Nearest-ball selection for ball collection.

COLLECTION ORDER (enforced):
  1. All white balls first (nearest from current robot position, re-evaluated each SEEK)
  2. Orange ball last (only targeted after all whites are collected)
  3. Then the state machine transitions to DRIVE_GOAL

The state machine calls get_target() on every SEEK entry to always pick
the closest remaining ball. advance() is a no-op kept for interface compat.
"""

import math
from dataclasses import dataclass
from typing import Optional

from golfbot_logger import get_logger

log = get_logger(__name__)


@dataclass
class RouteTarget:
    cm:      tuple   # world position in cm  — for angle maths
    px:      tuple   # pixel position this frame — for drive distance
    dist_px: float   # distance from robot_px to px


class RouteManager:

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_target(self, robot_pos: tuple, robot_px: tuple, world: dict) -> Optional[RouteTarget]:
        """
        Return the nearest RouteTarget, re-evaluated fresh each call.
        """
        white_balls, white_balls_px = _gather_white(world)
        orange_cm  = world.get("ob")
        orange_px  = world.get("ob_px")

        # ── Phase 1: white balls — always pick the nearest one ───────────
        if white_balls:
            idx = min(
                range(len(white_balls)),
                key=lambda i: _dist(robot_px, white_balls_px[i]) if robot_px and white_balls_px else float("inf"),
            )
            target_cm = white_balls[idx]
            target_px = white_balls_px[idx] if white_balls_px else None
            dist_px   = _dist(robot_px, target_px) if robot_px and target_px else 0.0
            log.debug("Nearest white ball: idx=%d  dist=%.0f px", idx, dist_px)
            return RouteTarget(cm=target_cm, px=target_px, dist_px=dist_px)

        # ── Phase 2: orange ball (all whites collected) ──────────────────
        if orange_cm is not None and orange_px is not None:
            dist_px = _dist(robot_px, orange_px) if robot_px else 0.0
            log.info("All white balls collected — going for the orange ball")
            return RouteTarget(cm=orange_cm, px=orange_px, dist_px=dist_px)

        # Nothing left
        return None

    def advance(self):
        """No-op: nearest-ball selection needs no route list to advance."""
        pass

    def clear(self):
        """No-op: no cached route to clear."""
        pass


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _gather_white(world: dict) -> tuple:
    """Return (white_balls_cm, white_balls_px) from the world dict."""
    return (
        world.get("white_balls", []),
        world.get("white_balls_px", []),
    )


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
