"""
Main loop for the Golfbot.

Every frame we:
1. Undistort the camera image
2. Warp the field to top-down view
3. Find the robot using ArUco
4. Find balls and cross using YOLO
5. Send everything to the state machine to get the next motor command
6. Draw the debug UI
"""

import time

import cv2

from golfbot_logger import setup_logging, get_logger

from vision.camera      import open_stream
from vision.field       import load_field_model, warp_field, detect_field
from vision.detector    import load_object_model, detect_objects, draw_debug_overlay
from vision.tracker     import WorldState, get_true_robot_pose, filter_detections_near_robot, build_world_state
from vision.aruco       import create_detector
from vision.lens_calibration import load_calibration, build_undistort_maps, undistort_frame

from controller.state_machine import GolfBotController
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

    # Lens calibration helps accuracy but isn't strictly required
    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        log.info("Lens calibration loaded, undistort maps built (%dx%d)", CAMERA_WIDTH, CAMERA_HEIGHT)

    # Open/close the claw once on startup before the main loop.
    log.info("Startup collect...")
    robot.reset_claw()

    controller    = GolfBotController()
    last_corners  = None
    last_no_field = 2.0   # when we last warned about missing field corners

    try:
        while True:
            # Get camera frame
            frame = stream.latest()
            if frame is None:
                continue                       # camera thread not ready yet
            frame = undistort_frame(frame, undist_maps)

            # Warp field to top-down
            last_corners = detect_field(field_model, frame, last_corners)
            if last_corners is None:
                now = time.time()
                if now - last_no_field > NO_FIELD_WARN_EVERY_S:
                    log.warning("Waiting for field corners...")
                    last_no_field = now
                if _show_and_wait(frame):      # keep the window responsive / allow ESC
                    break
                continue

            warped, homography = warp_field(frame, last_corners)
            h, w = warped.shape[:2]

            # Find robot
            robot_center, robot_angle = get_true_robot_pose(aruco_detector, frame, homography, w, h)

            # Find balls
            detections = detect_objects(object_model, warped)
            detections = filter_detections_near_robot(detections, robot_center)

            # Run state machine
            world:WorldState   = build_world_state(detections, robot_center, robot_angle, w, h)
            command = controller.update(world)

            # Update UI
            view  = controller.debug_view()
            debug = draw_debug_overlay(
                warped, detections, world, robot_center, robot_angle,
                view["state"], command.name,
                locked_target=view["target"],
                avoid_target=view["avoid_target"],
                next_waypoints=view["next_waypoints"],
            )
            cv2.imshow("GolfBot", debug)
            if (cv2.waitKey(1) & 0xFF) == 27:   # ESC
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        log.info("GolfBot stopped.")


def _show_and_wait(frame, ms=200):
    """Show a raw frame while waiting for the field; return True if ESC was pressed."""
    cv2.imshow("GolfBot", frame)
    return (cv2.waitKey(ms) & 0xFF) == 27


if __name__ == "__main__":
    main()
