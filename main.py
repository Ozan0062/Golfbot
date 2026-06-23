"""
main.py — GolfBot entry point.

Each camera frame, in order:
  1. Grab a frame and undistort lens distortion (if calibrated)
  2. Detect the field corners → perspective-warp to a top-down view
  3. Detect the robot pose (ArUco → warped → height-corrected)
  4. Detect balls and the cross (YOLO, on the warped frame)
  5. Build a "world" dict with every position in px and cm
  6. Feed the world into the state machine → get one Command back
  7. Draw the debug overlay and show it (ESC to quit)
"""

import time

import cv2

from golfbot_logger import setup_logging, get_logger

from vision.camera      import open_stream
from vision.field       import load_field_model, warp_field, detect_field
from vision.detector    import load_object_model, detect_objects, draw_debug_overlay
from vision.tracker     import get_true_robot_pose, filter_detections_near_robot, build_world_dict
from vision.aruco       import create_detector
from vision.lens_calibration import load_calibration, build_undistort_maps, undistort_frame

from controller.state_machine import GolfBotController
from controller.calibration_tracker import save_calibration_to_config
import controller.ev3_controller as robot
from config import CAMERA_WIDTH, CAMERA_HEIGHT

log = get_logger(__name__)

NO_FIELD_WARN_EVERY_S = 5.0   # throttle the "waiting for field" warning to once every N seconds


def main():
    setup_logging()
    log.info("GolfBot starting...")

    # Load models and open the camera.
    field_model    = load_field_model()
    object_model   = load_object_model()
    aruco_detector = create_detector()
    stream         = open_stream()

    # Lens calibration is optional — the pipeline works without it, just less accurately.
    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        log.info("Lens calibration loaded — undistort maps built (%dx%d)", CAMERA_WIDTH, CAMERA_HEIGHT)

    # Open/close the claw once on startup before the main loop.
    log.info("Startup collect...")
    robot.collect()

    controller    = GolfBotController()
    last_corners  = None
    last_no_field = 2.0   # when we last warned about missing field corners
    esc_pressed   = False  # did the user quit with ESC? -> save calibration on the way out

    try:
        while True:
            # 1. Grab and undistort the frame.
            frame = stream.latest()
            if frame is None:
                continue                       # camera thread not ready yet
            frame = undistort_frame(frame, undist_maps)

            # 2. Detect the field and warp to a top-down view.
            last_corners = detect_field(field_model, frame, last_corners)
            if last_corners is None:
                now = time.time()
                if now - last_no_field > NO_FIELD_WARN_EVERY_S:
                    log.warning("Waiting for field corners...")
                    last_no_field = now
                if _show_and_wait(frame):      # keep the window responsive / allow ESC
                    esc_pressed = True
                    break
                continue

            warped, homography = warp_field(frame, last_corners)
            h, w = warped.shape[:2]

            # 3. Robot pose (ArUco → warped → height-corrected).
            robot_center, robot_angle = get_true_robot_pose(aruco_detector, frame, homography, w, h)

            # 4. Balls and cross (YOLO on the warped frame).
            detections = detect_objects(object_model, warped)
            detections = filter_detections_near_robot(detections, robot_center)

            # 5 + 6. Build the world and run one state-machine tick.
            world   = build_world_dict(detections, robot_center, robot_angle, w, h)
            command = controller.update(world)

            # 7. Draw the overlay and show it.
            view  = controller.debug_view()
            debug = draw_debug_overlay(
                warped, detections, robot_center, robot_angle,
                view["state"], command.name,
                locked_target=view["target"],
                avoid_target=view["avoid_target"],
                next_waypoints=view["next_waypoints"],
            )
            cv2.imshow("GolfBot", debug)
            if (cv2.waitKey(1) & 0xFF) == 27:   # ESC
                esc_pressed = True
                break
    finally:
        # On ESC, persist the calibration learned this session as the new
        # starting values in config.py for the next run.
        if esc_pressed:
            try:
                px, deg_l, deg_r = save_calibration_to_config()
                log.info(
                    "Saved calibration to config — drive %.2f px/rot, turn L %.2f / R %.2f deg/rot",
                    px, deg_l, deg_r,
                )
            except Exception:
                log.exception("Failed to save calibration to config")
        stream.stop()
        cv2.destroyAllWindows()
        log.info("GolfBot stopped.")


def _show_and_wait(frame, ms=200):
    """Show a raw frame while waiting for the field; return True if ESC was pressed."""
    cv2.imshow("GolfBot", frame)
    return (cv2.waitKey(ms) & 0xFF) == 27


if __name__ == "__main__":
    main()
