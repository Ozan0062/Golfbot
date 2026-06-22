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

    def get_target(self, robot_pos: tuple, robot_px: tuple, world) -> Optional[RouteTarget]:
        """
        Return the nearest RouteTarget, re-evaluated fresh each call.
        """
        white_balls, white_balls_px = _gather_white(world)
        orange_cm  = world.ob
        orange_px  = world.ob_px

        # ── Phase 1: white balls — always pick the nearest one ───────────
        if white_balls:
            idx = min(
                range(len(white_balls)),
                key=lambda i: _dist(robot_px, white_balls_px[i]) if robot_px and white_balls_px else float("inf"),
            )
            target_cm_tuple = white_balls[idx]
            target_px_tuple = white_balls_px[idx] if white_balls_px else None
            target_cm = (target_cm_tuple[0], target_cm_tuple[1])
            target_px = (target_px_tuple[0], target_px_tuple[1]) if target_px_tuple else None
            dist_px   = _dist(robot_px, target_px) if robot_px and target_px else 0.0
            log.debug("Nearest white ball: idx=%d  dist=%.0f px", idx, dist_px)
            return RouteTarget(cm=target_cm, px=target_px, dist_px=dist_px)

        # ── Phase 2: orange ball (all whites collected) ──────────────────
        if orange_cm is not None and orange_px is not None:
            target_cm = (orange_cm[0], orange_cm[1])
            target_px = (orange_px[0], orange_px[1])
            dist_px = _dist(robot_px, target_px) if robot_px else 0.0
            log.info("All white balls collected — going for the orange ball")
            return RouteTarget(cm=target_cm, px=target_px, dist_px=dist_px)

        # Nothing left
        return None
    
    def get_target_dijkstras(self, path: list[dict], robot_px: tuple, world) -> Optional[RouteTarget]:
        """
        Returnerer den nærmeste RouteTarget direkte fra NetworkX/Dijkstra pathen.
        Den første ting i pathen er altid det bedste valg uanset type.
        """
        if not path:
            return None
            
        target_cm = path[0]["pos"]
        
        # Saml alle fysiske bolde for at finde det tilsvarende pixel-koordinat
        white_cm, white_px = _gather_white(world)
        all_cm = list(white_cm)
        all_px = list(white_px)
        
        if world.ob is not None and world.ob_px is not None:
            all_cm.append(world.ob)
            all_px.append(world.ob_px)
            
        target_px = None
        for i, ball_cm in enumerate(all_cm):
            # Find bolden via x og y match for at få dens pixels
            if math.isclose(ball_cm[0], target_cm[0], abs_tol=0.1) and math.isclose(ball_cm[1], target_cm[1], abs_tol=0.1):
                if i < len(all_px):
                    target_px = (all_px[i][0], all_px[i][1])
                break
                
        dist_px = _dist(robot_px, target_px) if robot_px and target_px else 0.0
        
        return RouteTarget(cm=target_cm, px=target_px, dist_px=dist_px)
    
    def advance(self):
        """No-op: nearest-ball selection needs no route list to advance."""
        pass

    def clear(self):
        """No-op: no cached route to clear."""
        pass


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _gather_white(world) -> tuple:
    """Return (white_balls_cm, white_balls_px) from the world state."""
    return (
        world.white_balls + world.white_wall_balls + world.white_corner_balls,
        world.white_balls_px + world.white_wall_balls_px + world.white_corner_balls_px,
    )


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
