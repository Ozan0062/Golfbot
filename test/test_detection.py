import math
import time
import cv2
import sys
import os
import numpy as np


# Ensure the parent directory (Golfbot) is at the start of sys.path to avoid stdlib name collisions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CAMERA_HEIGHT, CAMERA_WIDTH
from main import build_world_dict, detect_field, draw_debug_overlay, filter_detections_near_robot, undistort_frame
import state_detector
from vision.aruco import create_detector, detect_robot
from vision.calibration import build_undistort_maps, load_calibration
from vision.camera import open_stream
from vision.detector import detect_objects, load_object_model
from vision.field import load_field_model, warp_field


def detect_robot_pose_in_warped_coords(aruco_detector, raw_frame, homography_matrix):
    """
    Detect the ArUco marker on the raw (un-warped) frame, then project
    the robot's centre and heading into warped top-down coordinates.

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


if __name__ == "__main__":
    field_model    = load_field_model()
    object_model   = load_object_model()
    aruco_detector = create_detector()
    stream         = open_stream()
    
    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        print(f"Lens calibration loaded — undistort maps built ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    last_corners = None
    detector = state_detector.StateDetector()
    
    while True:
        # Stream latest frame and undistort
        frame = stream.latest()
        frame = undistort_frame(frame, undist_maps)
        if frame is None:
            continue   # camera thread not ready yet

        # Detect field and warp to top-down view 
        last_corners = detect_field(field_model, frame, last_corners)

        if last_corners is None:
            print("Waiting for field corners...")
            time.sleep(5) # Avoid spamming
            continue

        warped, homography = warp_field(frame, last_corners)
        h, w = warped.shape[:2]
        robot_center, robot_angle = detect_robot_pose_in_warped_coords(
        aruco_detector, frame, homography)
        
        # Detect objects
        detections = detect_objects(object_model, warped)
        detections = filter_detections_near_robot(detections, robot_center)
        
        world   = build_world_dict(detections, robot_center, robot_angle, w, h)
        command = detector.update(world)
        
        state_name = detector.state.name if hasattr(detector, 'state') else "UNKNOWN"
        command_name = command.name if command else "NONE"
        debug = draw_debug_overlay(warped, detections, robot_center, robot_angle, state_name, command_name)
        cv2.imshow("GolfBot", debug)
        if cv2.waitKey(1) & 0xFF == 27:
            break