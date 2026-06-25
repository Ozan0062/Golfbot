"""
tracker.py - convert pixel detections to real-world cm coordinates.

Full pipeline smoke test (camera -> field -> detect -> cm coords):
    python -m scripts.tracker_pipeline

Coordinate system: origin (0,0) is the top-left corner of the field.
X increases rightward, Y increases downward, both measured in cm.
"""

import math
import sys
from typing import Optional
sys.path.append(".")

import cv2
import numpy as np
from dataclasses import dataclass, field
from config import (FIELD_WIDTH_CM, FIELD_HEIGHT_CM,
                    CAMERA_CENTER_PX, CAMERA_HEIGHT_CM, ROBOT_MARKER_HEIGHT_CM,
                    WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
from controller.navigation import classify_zone
from vision.aruco import detect_robot
        
# calculates pixels to cm and saves it in variable "position_cm" in each object detection
def pixels_to_cm(detections, image_width, image_height,
                 field_w=FIELD_WIDTH_CM, field_h=FIELD_HEIGHT_CM):
    """
    Convert pixel coordinates from the warped field image to cm.
    Returns a new list of dicts with an added "position_cm" key: (x_cm, y_cm).
    """
    scale_x = field_w / image_width
    scale_y = field_h / image_height

    import copy
    results = []
    for det in detections:
        cx, cy = det.center
        det_copy = copy.copy(det)
        det_copy.position_cm = (cx * scale_x, cy * scale_y)
        results.append(det_copy)

    return results


def robot_px_to_cm(center_px, image_width, image_height,
                   field_w=FIELD_WIDTH_CM, field_h=FIELD_HEIGHT_CM):
    """
    Convert robot ArUco pixel position to cm.
    Returns (x_cm, y_cm) or None if center_px is None.
    """
    if center_px is None:
        return None
    scale_x = field_w / image_width
    scale_y = field_h / image_height
    return (center_px[0] * scale_x, center_px[1] * scale_y)


def detect_robot_pose_in_warped_coords(aruco_detector, raw_frame, homography_matrix):
    """Find robot in raw camera view and project to top-down view."""
    center_raw, angle_raw = detect_robot(aruco_detector, raw_frame)
    if center_raw is None:
        return None, None, None

    # Project two points: the marker centre and a point 50px ahead along the heading.
    # The angle between them in warped space gives the warped heading.
    forward_raw = (
        center_raw[0] + 50 * math.cos(math.radians(angle_raw)),
        center_raw[1] + 50 * math.sin(math.radians(angle_raw)),
    )
    pts = np.array([[center_raw, forward_raw]], dtype=np.float32)
    warped_pts = cv2.perspectiveTransform(pts, homography_matrix)[0]

    center  = (float(warped_pts[0][0]), float(warped_pts[0][1]))
    forward = (float(warped_pts[1][0]), float(warped_pts[1][1]))
    angle   = math.degrees(math.atan2(
        forward[1] - center[1],
        forward[0] - center[0],
    ))
    return center, forward, angle


def correct_robot_height(center_px, cam_center_px, cam_h, marker_h,
                         warped_w, warped_h, field_w, field_h):
    """
    Correct robot position for the QR marker sitting ~18 cm above the floor.

    The camera looks down at an angle, so a raised marker shows up shifted
    outward from the point directly below the camera. We undo that shift:
      1. Convert the marker's offset from the camera centre into cm.
      2. Angle from vertical: alpha = atan(d / H).
      3. Floor distance for a marker 18 cm up: d_actual = (H - 18) * tan(alpha).
      4. Scale the offset back inward toward the camera centre.
    """
    if center_px is None or cam_h <= marker_h or cam_h <= 0:
        return center_px

    sx = field_w / warped_w
    sy = field_h / warped_h

    dx_cm = (center_px[0] - cam_center_px[0]) * sx
    dy_cm = (center_px[1] - cam_center_px[1]) * sy

    d = math.hypot(dx_cm, dy_cm)

    if d < 0.5:
        return center_px

    alpha    = math.atan2(d, cam_h)
    d_actual = (cam_h - marker_h) * math.tan(alpha)
    scale    = d_actual / d

    return (
        cam_center_px[0] + (center_px[0] - cam_center_px[0]) * scale,
        cam_center_px[1] + (center_px[1] - cam_center_px[1]) * scale,
    )


def get_true_robot_pose(aruco_detector, raw_frame, homography_matrix,
                        warped_w, warped_h,
                        cam_center_px=CAMERA_CENTER_PX,
                        cam_h=CAMERA_HEIGHT_CM,
                        marker_h=ROBOT_MARKER_HEIGHT_CM,
                        field_w=FIELD_WIDTH_CM,
                        field_h=FIELD_HEIGHT_CM):
    """
    Full pipeline: detect ArUco -> project to warped coords -> correct for
    marker height -> recompute angle from corrected points.

    Returns (center_px, angle_deg) or (None, None).
    """
    center, forward, angle = detect_robot_pose_in_warped_coords(
        aruco_detector, raw_frame, homography_matrix
    )

    center = correct_robot_height(
        center, cam_center_px, cam_h, marker_h,
        warped_w, warped_h, field_w, field_h,
    )
    forward = correct_robot_height(
        forward, cam_center_px, cam_h, marker_h,
        warped_w, warped_h, field_w, field_h,
    )

    if center is not None and forward is not None:
        # Heading in the cm/floor frame, not the anisotropic warped-pixel frame:
        # scale the forward vector to cm per-axis before taking the angle, so the
        # reported heading is the robot's true physical heading at any orientation.
        sx = field_w / warped_w
        sy = field_h / warped_h
        angle = math.degrees(math.atan2(
            (forward[1] - center[1]) * sy,
            (forward[0] - center[0]) * sx,
        ))

    return center, angle


def filter_detections_near_robot(detections, robot_center_px, radius=None):
    """Ignore balls that are too close to the robot (usually false positives)."""
    if radius is None:
        from config import ROBOT_FILTER_RADIUS_PX
        radius = ROBOT_FILTER_RADIUS_PX
    if robot_center_px is None:
        return detections
    rx, ry = robot_center_px
    filtered = []
    for d in detections:
        dist = math.dist(d.center, (rx, ry))
        d.set_dist_from_robot(dist)
        if d.class_name not in ("wb", "ob") or dist > radius:
            filtered.append(d)
    return filtered


@dataclass
class WorldState:
    corners: list[tuple[float, float]] = field(default_factory=list)
    walls: list[tuple[float, float]] = field(default_factory=list)
    white_balls: list[tuple[float, float, str]] = field(default_factory=list)
    white_balls_px: list[tuple[float, float, str]] = field(default_factory=list)
    white_corner_balls: list[tuple[float, float, str]] = field(default_factory=list)
    white_corner_balls_px: list[tuple[float, float, str]] = field(default_factory=list)
    white_wall_balls: list[tuple[float, float, str]] = field(default_factory=list)
    white_wall_balls_px: list[tuple[float, float, str]] = field(default_factory=list)
    ob: Optional[tuple[float, float, str]] = None
    ob_px: Optional[tuple[float, float, str]] = None
    cross: Optional[tuple[float, float]] = None
    cross_px: Optional[tuple[float, float]] = None
    robot: Optional[tuple[float, float]] = None
    robot_px: Optional[tuple[float, float]] = None
    robot_angle: Optional[float] = None
    path:Optional[list[dict]] = None

def build_world_state(detections, robot_center, robot_angle, image_w, image_h) -> WorldState:
    """
    Combine YOLO detections and ArUco pose into a single world state.
    Contains both cm (for angle/bearing maths) and px (for drive distances).
    """
    world             = extract_objects(pixels_to_cm(detections, image_w, image_h))
    if world.cross_px is None:
        world.cross_px = (image_w / 2, image_h / 2)
        world.cross = robot_px_to_cm(world.cross_px, image_w, image_h)
    world.robot       = robot_px_to_cm(robot_center, image_w, image_h)
    world.robot_px    = robot_center
    world.robot_angle = robot_angle
    return world


def extract_objects(detections_cm) -> WorldState:
    """Sort YOLO bounding boxes into lists of white balls, orange ball, and cross."""
    objects = WorldState()
    best_ob_conf    = 0.0   # confidence of the orange ball kept so far
    best_cross_conf = 0.0   # confidence of the cross kept so far

    for det in detections_cm:
        name   = det.class_name
        pos_cm = det.position_cm
        pos_px = det.center          # original pixel coords from YOLO

        if name == "wb":
            zone, _ = classify_zone(pos_px, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
            ball_cm = (pos_cm[0], pos_cm[1], zone)
            ball_px = (pos_px[0], pos_px[1], zone)
            if zone == "corner":
                objects.white_corner_balls.append(ball_cm)
                objects.white_corner_balls_px.append(ball_px)
            elif zone == "wall":
                objects.white_wall_balls.append(ball_cm)
                objects.white_wall_balls_px.append(ball_px)
            else:
                objects.white_balls.append(ball_cm)
                objects.white_balls_px.append(ball_px)
        elif name == "ob" and (objects.ob is None or det.confidence > best_ob_conf):
            zone, _ = classify_zone(pos_px, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
            objects.ob = (pos_cm[0], pos_cm[1], zone)
            objects.ob_px = (pos_px[0], pos_px[1], zone)
            best_ob_conf = det.confidence
        elif name == "cross" and (objects.cross is None or det.confidence > best_cross_conf):
            objects.cross, objects.cross_px = pos_cm, pos_px
            best_cross_conf = det.confidence

    return objects
