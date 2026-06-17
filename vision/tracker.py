# vision/tracker.py — convert pixel detections to real-world cm coordinates
#
# Run for a full test (camera → field → detect → cm coords): python -m vision.tracker
# Coordinate system (cm):
#
# (0,0) ────── X ──────→ (180,0)
#   │                        │
#   │                        │
#   Y      (90,60)           │
#   │        center          │
#   │                        │
#   ↓                        │
# (0,120) ──────────── (180,120)
#
# Origin = top-left corner of field
# X increases rightward, Y increases downward


import math
import sys
sys.path.append(".")

import cv2
import numpy as np

from config import (FIELD_WIDTH_CM, FIELD_HEIGHT_CM,
                    CAMERA_CENTER_PX, CAMERA_HEIGHT_CM, ROBOT_MARKER_HEIGHT_CM)
from vision.aruco import detect_robot


def pixels_to_cm(detections, image_width, image_height,
                 field_w=FIELD_WIDTH_CM, field_h=FIELD_HEIGHT_CM):
    """
    Convert pixel coordinates from the warped field image to cm.
    Returns a new list of dicts with an added "position_cm" key: (x_cm, y_cm).
    """
    scale_x = field_w / image_width
    scale_y = field_h / image_height

    results = []
    for det in detections:
        cx, cy = det["center"]
        det_copy = dict(det)
        det_copy["position_cm"] = (cx * scale_x, cy * scale_y)
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
    """
    Detect the ArUco marker on the raw (un-warped) frame, then project
    the robot's centre and heading into warped top-down coordinates.

    Returns (center_px, forward_px, angle_deg) or (None, None, None).
    forward_px is needed so correct_robot_height can correct both points
    and recompute the angle — without this, the angle is off at field edges.
    """
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
    Correct robot position for the QR marker sitting 18 cm above the floor.

    Geometry (side view):

        Camera (C)
        |╲
        |  ╲          ← viewing angle α
        H    ╲
        |      ╲
        |  18cm QR ← marker up high
        |   |
        ────┴──── floor
     (312,303)  robot-base (what we want to find)

    1. Convert displacement from camera-center to cm
    2. Angle from vertical: α = atan(d / H)
    3. Horizontal distance to QR 18 cm up: d_actual = (H − 18) · tan(α)
    4. Scale displacement inward toward camera-center
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
    Full pipeline: detect ArUco → project to warped coords → correct for
    marker height → recompute angle from corrected points.

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
        angle = math.degrees(math.atan2(
            forward[1] - center[1],
            forward[0] - center[0],
        ))

    return center, angle


def filter_detections_near_robot(detections, robot_center_px, radius=None):
    """
    Remove ball detections whose pixel centre is within <radius> px of the robot's ArUco marker.
    """
    if radius is None:
        from config import ROBOT_FILTER_RADIUS_PX
        radius = ROBOT_FILTER_RADIUS_PX
    if robot_center_px is None:
        return detections
    rx, ry = robot_center_px
    return [
        d for d in detections
        if d["class_name"] not in ("wb", "ob")
        or math.dist(d["center"], (rx, ry)) > radius
    ]


def build_world_dict(detections, robot_center, robot_angle, image_w, image_h):
    """
    Combine YOLO detections and ArUco pose into a single world dict.
    Contains both cm (for angle/TSP maths) and px (for drive distances).
    """
    world                = extract_objects(pixels_to_cm(detections, image_w, image_h))
    world["robot"]       = robot_px_to_cm(robot_center, image_w, image_h)
    world["robot_px"]    = robot_center
    world["robot_angle"] = robot_angle
    return world


def extract_objects(detections_cm):
    """
    Split YOLO detections into named objects.
    Robot is NOT included here — it comes from ArUco separately.
    Returns dict with both cm and pixel positions:
        "cross":        (x_cm, y_cm) or None
        "cross_px":     (x_px, y_px) or None
        "ob":           (x_cm, y_cm) or None
        "white_balls":  [(x_cm, y_cm), ...]
        "ob_px":        (x_px, y_px) or None
        "white_balls_px": [(x_px, y_px), ...]
    """
    objects = {
        "cross":          None,
        "cross_px":       None,
        "ob":             None,
        "white_balls":    [],
        "ob_px":          None,
        "white_balls_px": [],
    }

    for det in detections_cm:
        name = det["class_name"]
        pos_cm = det["position_cm"]
        pos_px = det["center"]          # original pixel coords from YOLO

        if name == "wb":
            objects["white_balls"].append(pos_cm)
            objects["white_balls_px"].append(pos_px)
        elif name == "ob":
            if objects["ob"] is None or det["confidence"] > (objects.get("_ob_conf") or 0):
                objects["ob"]       = pos_cm
                objects["ob_px"]    = pos_px
                objects["_ob_conf"] = det["confidence"]
        elif name == "cross":
            if objects["cross"] is None or det["confidence"] > (objects.get("_cross_conf") or 0):
                objects["cross"]       = pos_cm
                objects["cross_px"]    = pos_px
                objects["_cross_conf"] = det["confidence"]

    return objects


# ── Standalone full pipeline test ───────────────────
if __name__ == "__main__":  # pragma: no cover - live camera/model smoke test
    from vision.camera import open_camera, grab_frame, release
    from vision.field import load_field_model, detect_corners, sort_corners, warp_field
    from vision.detector import load_object_model, detect_objects, draw_detections
    from vision.aruco import create_detector, detect_robot, draw_robot
    import cv2

    print("Loading models...")
    field_model = load_field_model()
    object_model = load_object_model()
    aruco_detector = create_detector()

    cap = open_camera()
    print("Grabbing frame...")
    frame = grab_frame(cap)
    release(cap)

    # Step 1: find field
    corners = detect_corners(field_model, frame)
    if len(corners) < 4:
        print(f"Only found {len(corners)} corners, need 4. Exiting.")
        sys.exit(1)

    sorted_c = sort_corners(corners)
    warped, M = warp_field(frame, sorted_c)
    h, w = warped.shape[:2]

    # Step 2: detect objects on warped image (YOLO)
    detections = detect_objects(object_model, warped)

    # Step 3: detect robot on warped image (ArUco)
    robot_center, robot_angle = detect_robot(aruco_detector, warped)

    # Step 4: convert to cm
    detections_cm = pixels_to_cm(detections, w, h)
    objects = extract_objects(detections_cm)

    robot_cm = robot_px_to_cm(robot_center, w, h)

    # Print results
    print(f"\n{'─' * 40}")
    print(f"Field warped to {w}x{h} px → {FIELD_WIDTH_CM}x{FIELD_HEIGHT_CM} cm")
    print(f"{'─' * 40}")

    if robot_cm:
        print(f"  robot:        ({robot_cm[0]:.1f}, {robot_cm[1]:.1f}) cm  heading={robot_angle:.0f}°")
    else:
        print(f"  robot:        not found (no ArUco marker detected)")

    for key, val in objects.items():
        if key == "white_balls":
            print(f"  white_balls ({len(val)}):")
            for i, pos in enumerate(val):
                print(f"    [{i}] ({pos[0]:.1f}, {pos[1]:.1f}) cm")
        elif val:
            print(f"  {key:12s}: ({val[0]:.1f}, {val[1]:.1f}) cm")
        else:
            print(f"  {key:12s}: not found")

    # Show annotated image
    display = draw_detections(warped, detections)
    display = draw_robot(display, robot_center, robot_angle)
    cv2.imshow("Full Pipeline", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
