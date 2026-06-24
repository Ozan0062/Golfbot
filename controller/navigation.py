"""
navigation.py — math helpers for the GolfBot controller.
"""
import math

from config import FIELD_WIDTH_CM, FIELD_HEIGHT_CM, WARPED_WIDTH, WARPED_HEIGHT

# Per-axis warped-pixel -> cm scale. The warp canvas (900x600) does not match the
# field's aspect ratio (170x124.5), so these two are NOT equal -- px space is
# anisotropic. Do all *angle* maths in cm to avoid heading-dependent distortion.
_PX_TO_CM_X = FIELD_WIDTH_CM / WARPED_WIDTH
_PX_TO_CM_Y = FIELD_HEIGHT_CM / WARPED_HEIGHT


def px_to_cm(point_px):
    """Convert a warped-image pixel point to field cm (per-axis scale)."""
    return (point_px[0] * _PX_TO_CM_X, point_px[1] * _PX_TO_CM_Y)


def px_angle_to_cm(angle_deg):
    """
    Re-express a pixel-frame heading as the equivalent physical (cm) heading.

    Identity for axis-aligned headings (0/+-90/180); only diagonals shift, because
    the anisotropic warp stretches x and y differently. Used so a staging point
    placed in pixel space and the heading we align to before driving in agree.
    """
    rad = math.radians(angle_deg)
    return math.degrees(math.atan2(_PX_TO_CM_Y * math.sin(rad),
                                   _PX_TO_CM_X * math.cos(rad)))


def angle_to_target(robot_pos, target_pos):
    """
    Bearing in degrees from robot_pos to target_pos. Feed cm coordinates (not
    raw warped pixels) so the bearing is undistorted by the anisotropic warp.
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

# Required robot heading (deg) for a ball touching a single wall, by wall name.
_WALL_APPROACH_ANGLE = {
    "top":    -90.0,
    "bottom":  90.0,
    "left":   180.0,
    "right":    0.0,
}

# Required robot heading (deg) for a ball in a corner, keyed by the wall pair.
_CORNER_APPROACH_ANGLE = {
    frozenset({"top", "left"}):     -135.0,
    frozenset({"top", "right"}):     -45.0,
    frozenset({"bottom", "left"}):   135.0,
    frozenset({"bottom", "right"}):   45.0,
}


def wall_approach_angle(walls):
    """
    Required robot heading (degrees) when collecting a wall or corner ball.

    Wall balls: perpendicular to the wall (robot drives straight into it).
    Corner balls: 45 deg diagonal into the corner.

    Uses the same angle convention as angle_to_target:
      0 deg = right, 90 deg = down, -90 deg = up, 180 deg = left.

    Returns None for open-field balls (no constraint).
    """
    if len(walls) >= 2:
        return _CORNER_APPROACH_ANGLE.get(frozenset(walls[:2]))
    if len(walls) == 1:
        return _WALL_APPROACH_ANGLE.get(walls[0])
    return None


# --- Cross approach geometry ----------------------------------------------

def cross_approach_angle(ball_px, cross_px):
    """
    Fixed-diagonal heading (deg) for collecting a ball at the centre cross.

    Computes the direction from cross centre → ball, adds 180° to get the
    approach direction (robot drives from the ball's side toward the cross),
    then snaps to the nearest 45° diagonal.  This is more robust than a strict
    quadrant comparison: detection noise must shift the angle by >22.5° before
    a wrong diagonal is chosen.
    """
    dx = ball_px[0] - cross_px[0]
    dy = ball_px[1] - cross_px[1]
    raw = math.degrees(math.atan2(dy, dx))   # direction cross→ball
    approach = raw + 180.0                   # robot faces the opposite way
    # Snap only to the diagonals: 45, 135, -45, -135 (gaps between cross legs)
    snapped = round((approach - 45.0) / 90.0) * 90.0 + 45.0
    # Normalise to (-180, 180]
    while snapped >  180.0: snapped -= 360.0
    while snapped <= -180.0: snapped += 360.0
    return snapped


def cross_trigger_radius(radius_px):
    """
    Distance from the cross centre within which a ball is treated as a cross ball.
    Returns 2x the cross radius.
    """
    return radius_px * 2


def staging_point(target_px, approach_angle_deg, standoff_px):
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
                      field_width, field_height, prefer_near_px=None):
    """
    Compute a waypoint to steer around a blocking obstacle.

    Places the waypoint perpendicular to the robot→target line at the obstacle's
    projection.  Both candidates (left and right of the path) are evaluated;
    the one closest to `prefer_near_px` (defaults to robot_px) is preferred.
    Falls back to the other side if the preferred candidate is outside the field.

    Returns (x, y) waypoint in pixel coordinates, or None if the path
    has zero length.
    """
    dx = target_px[0] - robot_px[0]
    dy = target_px[1] - robot_px[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None

    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux   # perpendicular unit vector (left of path)

    t = (obstacle_px[0] - robot_px[0]) * ux + (obstacle_px[1] - robot_px[1]) * uy
    proj = (robot_px[0] + t * ux, robot_px[1] + t * uy)

    # Pull the waypoint back along the path so the diagonal approach clears the obstacle
    t_pullback = max(0, t - clearance_px)
    proj_pullback = (robot_px[0] + t_pullback * ux, robot_px[1] + t_pullback * uy)

    wp_left  = (proj_pullback[0] + nx * clearance_px, proj_pullback[1] + ny * clearance_px)
    wp_right = (proj_pullback[0] - nx * clearance_px, proj_pullback[1] - ny * clearance_px)

    def in_bounds(wp):
        return (clearance_px <= wp[0] <= field_width  - clearance_px and
                clearance_px <= wp[1] <= field_height - clearance_px)

    def clamp(wp):
        return (max(clearance_px, min(wp[0], field_width  - clearance_px)),
                max(clearance_px, min(wp[1], field_height - clearance_px)))

    ref = prefer_near_px if prefer_near_px is not None else robot_px
    d_left  = math.hypot(wp_left[0]  - ref[0], wp_left[1]  - ref[1])
    d_right = math.hypot(wp_right[0] - ref[0], wp_right[1] - ref[1])

    if d_left <= d_right:
        preferred, fallback = wp_left, wp_right
    else:
        preferred, fallback = wp_right, wp_left

    wp = preferred if in_bounds(preferred) else fallback
    return clamp(wp)


# --- Cross-obstacle geometry -------------------------------------------------

def cross_avoid_points(cross_px, field_width, field_height):
    """
    Eight fixed navigation waypoints around the cross: four cardinal and four
    diagonal. Each is placed at the midpoint between the cross centre and the
    corresponding wall (cardinal) or field corner (diagonal). The diagonals let
    the dodge planner route around a corner when the robot and target sit in
    diagonally-opposite quadrants - a case the 4 cardinal points alone can't
    express. Keyed by heading angle (0=right, 45=down-right, 90=down,
    135=down-left, 180=left, 225=up-left, 270=up, 315=up-right).
    """
    cx, cy = cross_px
    return {
          0: ((cx + field_width)  / 2, cy),                       # right
         45: ((cx + field_width)  / 2, (cy + field_height) / 2),  # down-right
         90: (cx, (cy + field_height) / 2),                       # down
        135: (cx / 2,             (cy + field_height) / 2),       # down-left
        180: (cx / 2,             cy),                            # left
        225: (cx / 2,             cy / 2),                        # up-left
        270: (cx, cy / 2),                                        # up
        315: ((cx + field_width)  / 2, cy / 2),                   # up-right
    }
