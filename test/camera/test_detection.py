import time
import cv2
import sys
import os


# Ensure the project root (Golfbot) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CAMERA_HEIGHT, CAMERA_WIDTH
from test.camera.state_detector import StateDetector
from vision.aruco import create_detector
from vision.lens_calibration import build_undistort_maps, load_calibration, undistort_frame
from vision.camera import open_stream
from vision.detector import detect_objects, load_object_model, draw_debug_overlay
from vision.field import load_field_model, warp_field, detect_field
from vision.tracker import get_true_robot_pose, filter_detections_near_robot, build_world_dict


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
    detector = StateDetector()

    while True:
        # Stream latest frame and undistort
        frame = stream.latest()
        if frame is None:
            continue   # camera thread not ready yet
        frame = undistort_frame(frame, undist_maps)

        # Detect field and warp to top-down view
        last_corners = detect_field(field_model, frame, last_corners)

        if last_corners is None:
            print("Waiting for field corners...")
            time.sleep(5) # Avoid spamming
            continue

        warped, homography = warp_field(frame, last_corners)
        h, w = warped.shape[:2]

        # Detect robot pose (ArUco → warped → height-corrected)
        robot_center, robot_angle = get_true_robot_pose(
            aruco_detector, frame, homography, w, h
        )

        # Detect objects
        detections = detect_objects(object_model, warped)
        detections = filter_detections_near_robot(detections, robot_center)

        world   = build_world_dict(detections, robot_center, robot_angle, w, h)
        command = detector.update(world)

        state_name = detector.state.name if hasattr(detector, 'state') else "UNKNOWN"
        command_name = command.name if command else "NONE"
        debug = draw_debug_overlay(warped, detections, robot_center, robot_angle, state_name, command_name)
        cv2.imshow("GolfBot", debug)
        # Wait for space to advance, ESC to quit
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:       # ESC
                stream.stop()
                cv2.destroyAllWindows()
                sys.exit(0)
            if key == ord(' '):  # Space → next frame
                break
