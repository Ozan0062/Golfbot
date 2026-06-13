"""
main.py — GolfBot entry point.

Main loop:
  1. Grab a camera frame
  2. Undistort lens distortion (if calibrated)
  3. Detect field corners → perspective-warp to top-down view
  4. Detect robot pose via ArUco marker (on raw frame, projected into warped coords)
  5. Detect balls and obstacles via YOLO (on warped frame)
  6. Build a "world" dict with all positions in both px and cm
  7. Feed world into the state machine → get a Command back
  8. Draw debug overlay and display
"""

import math
import cv2
import numpy as np

from vision.camera   import open_stream
from vision.field    import load_field_model, detect_corners, sort_corners, warp_field
from vision.detector import load_object_model, detect_objects, draw_detections
from vision.tracker  import pixels_to_cm, extract_objects, robot_px_to_cm
from vision.aruco    import create_detector, detect_robot, draw_robot
from vision.calibration import load_calibration, build_undistort_maps, remap

from controller.state_machine import GolfBotController
from config import ROBOT_FILTER_RADIUS_PX, CAMERA_WIDTH, CAMERA_HEIGHT


# ─── Vision pipeline helpers ─────────────────────────────────────────────────

def undistort_frame(frame, undist_maps):
    """Apply lens correction if calibration data is available."""
    if undist_maps is not None:
        return remap(frame, *undist_maps)
    return frame


def detect_field(field_model, frame, last_corners):
    """
    Detect field corners in the current frame.
    Returns updated corners (or the previous ones if detection fails this frame).

    DECISION: We keep last_corners as a fallback so one bad frame doesn't
    kill the pipeline. The field doesn't move, so stale corners are fine.
    """
    corners = detect_corners(field_model, frame)
    if len(corners) >= 4:
        return sort_corners(corners)
    return last_corners


def detect_robot_pose_in_warped_coords(aruco_detector, raw_frame, homography_matrix):
    """
    Detect the ArUco marker on the raw (un-warped) frame, then project
    the robot's centre and heading into warped top-down coordinates.

    DECISION: We detect ArUco on the raw frame (not warped) because the
    warp can distort the marker enough to break detection. The centre and
    a forward-pointing reference point are then projected through the
    homography so all downstream code works in warped-pixel space.

    Returns (center_px, angle_deg) or (None, None).
    """
    center_raw, angle_raw = detect_robot(aruco_detector, raw_frame)
    if center_raw is None:
        return None, None

    # Project two points: the marker centre and a point 50px ahead along the heading.
    # The angle between them in warped space gives the warped heading.
    forward_raw = (
        center_raw[0] + 50 * math.cos(math.radians(angle_raw)),
        center_raw[1] + 50 * math.sin(math.radians(angle_raw)),
    )
    pts = np.array([[center_raw, forward_raw]], dtype=np.float32)
    warped_pts = cv2.perspectiveTransform(pts, homography_matrix)[0]

    center = (float(warped_pts[0][0]), float(warped_pts[0][1]))
    angle  = math.degrees(math.atan2(
        warped_pts[1][1] - warped_pts[0][1],
        warped_pts[1][0] - warped_pts[0][0],
    ))
    return center, angle


def filter_detections_near_robot(detections, robot_center_px, radius=ROBOT_FILTER_RADIUS_PX):
    """
    Remove ball detections whose pixel centre is within <radius> px of the
    robot's ArUco marker.

    DECISION: The YOLO model sometimes detects the ArUco marker or parts
    of the robot body as a ball. Filtering by proximity to the known
    robot position eliminates these false positives.
    """
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


def draw_debug_overlay(warped, detections, robot_center, robot_angle, state_name, command_name):
    """Draw detections, robot marker, and current state on the frame."""
    debug = draw_detections(warped, detections)
    debug = draw_robot(debug, robot_center, robot_angle)
    cv2.putText(debug, f"{state_name}  {command_name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return debug


# ─── Main loop ───────────────────────────────────────────────────────────────

def main():
    print("GolfBot starting...")

    # Load models and open camera
    field_model    = load_field_model()
    object_model   = load_object_model()
    aruco_detector = create_detector()
    stream         = open_stream()

    # Load lens calibration (optional — works without it, just less accurate)
    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        print(f"Lens calibration loaded — undistort maps built ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")

    controller   = GolfBotController()
    last_corners = None

    while True:
        # ── 1. Grab and undistort frame ──────────────────────────────────
        frame = stream.latest()
        if frame is None:
            continue   # camera thread not ready yet

        frame = undistort_frame(frame, undist_maps)

        # ── 2. Detect field and warp to top-down view ────────────────────
        last_corners = detect_field(field_model, frame, last_corners)
        if last_corners is None:
            print("Waiting for field corners...")
            cv2.imshow("GolfBot", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        warped, homography = warp_field(frame, last_corners)
        h, w = warped.shape[:2]

        # ── 3. Detect robot pose (ArUco on raw frame → warped coords) ───
        robot_center, robot_angle = detect_robot_pose_in_warped_coords(
            aruco_detector, frame, homography
        )

        # ── 4. Detect balls and obstacles (YOLO on warped frame) ─────────
        detections = detect_objects(object_model, warped)
        detections = filter_detections_near_robot(detections, robot_center)

        # ── 5. Build world dict and run state machine ────────────────────
        world   = build_world_dict(detections, robot_center, robot_angle, w, h)
        command = controller.update(world)

        # ── 6. Debug overlay ─────────────────────────────────────────────
        debug = draw_debug_overlay(warped, detections, robot_center, robot_angle,
                                   controller.state.name, command.name)
        cv2.imshow("GolfBot", debug)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    stream.stop()


if __name__ == "__main__":
    main()
