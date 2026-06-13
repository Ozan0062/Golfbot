"""
route_manager.py — TSP route state for ball collection.

Owns:
  • gathering visible balls from the world dict
  • deciding when to (re)compute the Christofides route
  • looking up the current pixel position of the next target each frame
  • advancing or clearing the route

The state machine calls get_target() each frame and advance() after a collect.
It never touches _ball_route directly.
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
        self._route:       list = []   # ordered cm positions
        self._ball_count:  int  = 0    # #balls when route was last computed

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_target(self, robot_pos: tuple, robot_px: tuple, world: dict) -> Optional[RouteTarget]:
        """
        Return the next RouteTarget, or None if no balls are visible.

        Recomputes the route when new balls appear that weren't in the plan.
        Px positions are refreshed from the current frame every call.
        """
        all_balls, all_balls_px = _gather_balls(world)

        if not all_balls:
            return None

        if not self._route or len(all_balls) > self._ball_count:
            self._compute(robot_pos, all_balls)

        target_cm = self._route[0]
        target_px = _nearest_px(target_cm, all_balls, all_balls_px)
        dist_px   = _dist(robot_px, target_px) if robot_px and target_px else 0.0

        return RouteTarget(cm=target_cm, px=target_px, dist_px=dist_px)

    def advance(self):
        """Call after a successful collect to move to the next target."""
        if self._route:
            self._route.pop(0)
        remaining = len(self._route)
        print(f"[ROUTE] Advanced.  {remaining} target(s) remaining.")

    def clear(self):
        """Call when no balls are visible so the route is rebuilt fresh next time."""
        self._route      = []
        self._ball_count = 0

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _compute(self, robot_pos: tuple, all_balls: list):
        points = [robot_pos] + all_balls          # index 0 = robot
        order  = christofides_route(points)

        self._route = [
            all_balls[i - 1]
            for i in order
            if i != 0 and 1 <= i <= len(all_balls)
        ]
        self._ball_count = len(all_balls)
        print(f"[ROUTE] Christofides over {len(all_balls)} ball(s) → "
              f"order {[i for i in order if i != 0]}")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _gather_balls(world: dict) -> tuple:
    """Return (all_balls_cm, all_balls_px) from the world dict."""
    white    = world.get("white_balls", [])
    white_px = world.get("white_balls_px", [])
    orange   = world.get("ob")
    orange_px = world.get("ob_px")

    balls    = list(white)
    balls_px = list(white_px)
    if orange:
        balls.append(orange)
        balls_px.append(orange_px)

    return balls, balls_px


def _nearest_px(target_cm: tuple, all_balls: list, all_balls_px: list) -> Optional[tuple]:
    """Return the pixel position of the ball whose cm position is closest to target_cm."""
    if not all_balls:
        return None
    idx = min(range(len(all_balls)), key=lambda i: _dist(all_balls[i], target_cm))
    return all_balls_px[idx]


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
