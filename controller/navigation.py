"""
navigation.py — math helpers for the GolfBot controller.
"""
import math


def angle_to_target(robot_pos, target_pos):
    """degrees from robot_pos to target_pos."""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]   # NOTE: y increases downward in image coords
    return math.degrees(math.atan2(dy, dx))


def angle_error(current_angle, desired_angle):
    """Shortest signed angle from current_angle to desired_angle.
    Positive = clockwise (RIGHT), Negative = counter-clockwise (LEFT).
    """
    return (desired_angle - current_angle + 180) % 360 - 180


def nearest_ball(robot_pos, balls):
    """Return the position of the closest ball (white or orange), or None."""
    if not balls:
        return None
    return min(balls, key=lambda b: math.dist(robot_pos, b))


def distance(pos_a, pos_b):
    """distance between two points in cm."""
    return math.dist(pos_a, pos_b)
