"""
navigation.py — math helpers for the GolfBot controller.
"""
import math


def angle_to_target(robot_pos, target_pos):
    """
    Bearing in degrees from robot_pos to target_pos.
    """
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    return math.degrees(math.atan2(dy, dx))


def angle_error(current_angle, desired_angle):
    """
    Shortest signed angle from current_angle to desired_angle.
    """
    return (desired_angle - current_angle + 180) % 360 - 180


def cm_to_pixels(pos_cm, image_width, image_height, field_width, field_height):
    """
    Convert a field position in cm to warped-image pixel coordinates.
    """
    return (
        pos_cm[0] * image_width / field_width,
        pos_cm[1] * image_height / field_height,
    )


# --------------------------------------------------------------------------
# Path planning helpers
# --------------------------------------------------------------------------


# --- Zone classification -------------------------------------------------

def classify_zone(target_px, wall_margin, field_width, field_height):
    """
    Determine if a target is in open field, near a wall, or in a corner.

    Returns (zone, walls) where:
      zone  -- "open", "wall", or "corner"
      walls -- list of nearby wall names: "top", "bottom", "left", "right"
    """
    walls = []
    if target_px[0] < wall_margin:
        walls.append("left")
    if target_px[0] > field_width - wall_margin:
        walls.append("right")
    if target_px[1] < wall_margin:
        walls.append("top")
    if target_px[1] > field_height - wall_margin:
        walls.append("bottom")

    if len(walls) >= 2:
        return "corner", walls
    if len(walls) == 1:
        return "wall", walls
    return "open", []


# --- Wall / corner approach geometry --------------------------------------

def wall_approach_angle(walls):
    """
    Required robot heading (degrees) when collecting a wall or corner ball.

    Wall balls: perpendicular to the wall (robot drives straight into it).
    Corner balls: 45 deg diagonal into the corner.

    Uses the same angle convention as angle_to_target:
      0 deg = right, 90 deg = down, -90 deg = up, 180 deg = left.

    Returns None for open-field balls (no constraint).
    """
    _WALL = {
        "top":    -90.0,
        "bottom":  90.0,
        "left":   180.0,
        "right":    0.0,
    }

    _CORNER = {
        frozenset({"top", "left"}):     -135.0,
        frozenset({"top", "right"}):     -45.0,
        frozenset({"bottom", "left"}):   135.0,
        frozenset({"bottom", "right"}):   45.0,
    }

    if len(walls) >= 2:
        return _CORNER.get(frozenset(walls[:2]))
    if len(walls) == 1:
        return _WALL.get(walls[0])
    return None


def staging_point(target_px, approach_angle_deg, standoff_px):
    """
    Position the robot should reach before the final straight-line approach.

    Placed standoff_px pixels away from the target, directly behind the
    approach angle. From here the robot just drives straight to collect.

    Example: ball near top wall, approach angle = -90 deg (heading up).
      staging = (ball_x, ball_y + standoff) -- robot waits below the ball.
    """
    angle_rad = math.radians(approach_angle_deg)
    return (
        target_px[0] - standoff_px * math.cos(angle_rad),
        target_px[1] - standoff_px * math.sin(angle_rad),
    )


# --- Obstacle avoidance ---------------------------------------------------

def _point_to_segment_dist(point, seg_a, seg_b):
    """Shortest distance from a point to a line segment A->B."""
    ax, ay = seg_a
    bx, by = seg_b
    px, py = point

    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def path_is_clear(start_px, end_px, obstacles, clearance_px):
    """
    Check if a straight-line path avoids all obstacles.

    obstacles -- list of (x, y) pixel positions of objects on the field.
    clearance_px -- minimum allowed distance from path to any obstacle.

    Returns (clear, blocker):
      clear   -- True if nothing is within clearance_px of the path.
      blocker -- the (x, y) of the nearest blocking obstacle, or None.
    """
    nearest = None
    nearest_dist = float("inf")

    for obs in obstacles:
        dist = _point_to_segment_dist(obs, start_px, end_px)
        if dist < clearance_px and dist < nearest_dist:
            nearest_dist = dist
            nearest = obs

    return (nearest is None, nearest)


def obstacle_waypoint(robot_px, target_px, obstacle_px, clearance_px,
                      field_width, field_height):
    """
    Compute a waypoint to steer around a blocking obstacle.

    The waypoint is placed alongside the obstacle (same progress along the
    path) but offset perpendicular to the path on the opposite side from
    the obstacle. If the opposite side is too close to a wall, it flips
    to the same side but further out.

    Returns (x, y) waypoint in pixel coordinates, or None if the path
    has zero length.
    """
    dx = target_px[0] - robot_px[0]
    dy = target_px[1] - robot_px[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None

    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    t = (obstacle_px[0] - robot_px[0]) * ux + (obstacle_px[1] - robot_px[1]) * uy
    proj = (robot_px[0] + t * ux, robot_px[1] + t * uy)

    side = (obstacle_px[0] - proj[0]) * nx + (obstacle_px[1] - proj[1]) * ny

    if side >= 0:
        wp = (proj[0] - nx * clearance_px, proj[1] - ny * clearance_px)
    else:
        wp = (proj[0] + nx * clearance_px, proj[1] + ny * clearance_px)

    if (wp[0] < clearance_px or wp[0] > field_width - clearance_px or
            wp[1] < clearance_px or wp[1] > field_height - clearance_px):
        if side >= 0:
            wp = (proj[0] + nx * clearance_px, proj[1] + ny * clearance_px)
        else:
            wp = (proj[0] - nx * clearance_px, proj[1] - ny * clearance_px)

    wp = (
        max(clearance_px, min(wp[0], field_width - clearance_px)),
        max(clearance_px, min(wp[1], field_height - clearance_px)),
    )

    return wp
