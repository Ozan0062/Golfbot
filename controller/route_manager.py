"""
route_manager.py — TSP route state for ball collection.

COLLECTION ORDER (enforced):
  1. All white balls first (Christofides-optimized order)
  2. Orange ball last (only targeted after all whites are collected)
  3. Then the state machine transitions to DRIVE_GOAL

The state machine calls get_target() once per SEEK entry and advance()
after each collect. It never touches _route directly.
"""

import math
from dataclasses import dataclass
from typing import Optional

from controller.tsp_christofides import christofides_route


@dataclass
class RouteTarget:
    cm:      tuple   # world position in cm  — for angle maths
    px:      tuple   # pixel position this frame — for drive distance
    dist_px: float   # distance from robot_px to px


class RouteManager:

    def __init__(self):
        self._route:       list = []   # ordered cm positions (white balls only)
        self._white_count: int  = 0    # white balls when route was last computed

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_target(self, robot_pos: tuple, robot_px: tuple, world: dict) -> Optional[RouteTarget]:
        """
        Return the next RouteTarget, or None if no balls remain.

        DECISION: White balls are always collected first. Christofides runs
        only on white balls. The orange ball is returned as the target only
        after all whites are collected. This guarantees the intended order:
        whites → orange → goal.
        """
        white_balls, white_balls_px = _gather_white(world)
        orange_cm  = world.get("ob")
        orange_px  = world.get("ob_px")

        # ── Phase 1: white balls ─────────────────────────────────────────
        if white_balls:
            if not self._route or len(white_balls) > self._white_count:
                self._compute_white_route(robot_pos, white_balls)

            target_cm = self._route[0]
            target_px = _nearest_px(target_cm, white_balls, white_balls_px)
            dist_px   = _dist(robot_px, target_px) if robot_px and target_px else 0.0
            return RouteTarget(cm=target_cm, px=target_px, dist_px=dist_px)

        # ── Phase 2: orange ball (all whites collected) ──────────────────
        if orange_cm is not None and orange_px is not None:
            dist_px = _dist(robot_px, orange_px) if robot_px else 0.0
            print("[ROUTE] All whites collected — targeting orange ball")
            return RouteTarget(cm=orange_cm, px=orange_px, dist_px=dist_px)

        # Nothing left
        return None

    def advance(self):
        """Call after a successful collect to move to the next target."""
        if self._route:
            self._route.pop(0)
        remaining = len(self._route)
        print(f"[ROUTE] Advanced.  {remaining} white target(s) remaining.")

    def clear(self):
        """Call when no balls are visible so the route is rebuilt fresh next time."""
        self._route       = []
        self._white_count = 0

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _compute_white_route(self, robot_pos: tuple, white_balls: list):
        """Run Christofides on white balls only. Orange is handled separately."""
        points = [robot_pos] + white_balls     # index 0 = robot
        order  = christofides_route(points)

        self._route = [
            white_balls[i - 1]
            for i in order
            if i != 0 and 1 <= i <= len(white_balls)
        ]
        self._white_count = len(white_balls)
        print(f"[ROUTE] Christofides over {len(white_balls)} white ball(s) → "
              f"order {[i for i in order if i != 0]}")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _gather_white(world: dict) -> tuple:
    """Return (white_balls_cm, white_balls_px) from the world dict."""
    return (
        world.get("white_balls", []),
        world.get("white_balls_px", []),
    )


def _nearest_px(target_cm: tuple, all_balls: list, all_balls_px: list) -> Optional[tuple]:
    """Return the pixel position of the ball whose cm position is closest to target_cm."""
    if not all_balls:
        return None
    idx = min(range(len(all_balls)), key=lambda i: _dist(all_balls[i], target_cm))
    return all_balls_px[idx]


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
