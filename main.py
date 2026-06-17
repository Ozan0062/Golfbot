"""
main.py — GolfBot entry point.

Main loop:
  1. Grab a camera frame
  2. Undistort lens distortion (if calibrated)
  3. Detect field corners → perspective-warp to top-down view
  4. Detect robot pose (ArUco → warped → height-corrected)
  5. Detect balls and obstacles via YOLO (on warped frame)
  6. Build a "world" dict with all positions in both px and cm
  7. Feed world into the state machine → get a Command back
  8. Draw debug overlay and display
"""

import cv2
import time

from golfbot_logger import setup_logging, get_logger

from vision.camera      import open_stream
from vision.field       import load_field_model, warp_field, detect_field
from vision.detector    import load_object_model, detect_objects, draw_debug_overlay
from vision.tracker     import get_true_robot_pose, filter_detections_near_robot, build_world_dict
from vision.aruco       import create_detector
from vision.calibration import load_calibration, build_undistort_maps, undistort_frame

from controller.state_machine import GolfBotController
import controller.ev3_controller as robot
from config import CAMERA_WIDTH, CAMERA_HEIGHT

log = get_logger(__name__)


def main():
    setup_logging(level="INFO")
    log.info("GolfBot starting...")

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
        log.info("Lens calibration loaded — undistort maps built (%dx%d)", CAMERA_WIDTH, CAMERA_HEIGHT)

    # Run collect once on startup (open/close claw before main loop)
    log.info("Startup collect...")
    robot.collect()

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
            log.warning("Waiting for field corners...")
            time.sleep(5) # Avoid spamming
            continue

        warped, homography = warp_field(frame, last_corners)
        h, w = warped.shape[:2]

        # ── 3. Detect robot pose (ArUco → warped → height-corrected) ────
        robot_center, robot_angle = get_true_robot_pose(
            aruco_detector, frame, homography, w, h
        )

        # ── 4. Detect balls and obstacles (YOLO on warped frame) ─────────
        detections = detect_objects(object_model, warped)
        detections = filter_detections_near_robot(detections, robot_center)

        # ── 5. Build world dict and run state machine ────────────────────
        world   = build_world_dict(detections, robot_center, robot_angle, w, h)
        command = controller.update(world)

        # ── 6. Debug overlay ─────────────────────────────────────────────
        debug = draw_debug_overlay(warped, detections, robot_center, robot_angle,
                                   controller.state.name, command.name, controller._locked_target)
        cv2.imshow("GolfBot", debug)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    stream.stop()


if __name__ == "__main__":
    main()