"""
calibrate.py — standalone drive/turn calibration tool.

Run this before main.py to get fresh calibration values. The script drives
the robot forward, turns left, then turns right, measures each move via the
overhead camera, and saves the results to config.py.

Usage:
    python calibrate.py

Press C in the camera window to start, any other key to quit without saving.
"""

import time

import cv2

from golfbot_logger import setup_logging, get_logger
from vision.camera          import open_stream
from vision.field           import load_field_model, warp_field, detect_field
from vision.tracker         import get_true_robot_pose
from vision.aruco           import create_detector
from vision.lens_calibration import load_calibration, build_undistort_maps, undistort_frame
from controller.calibration_tracker import (
    save_calibration_to_config,
    calibration_pixels, calibration_angle_left, calibration_angle_right,
)
from controller.drive_calibration import measure_pixels_per_rotation, measure_degrees_per_rotation
import controller.ev3_controller as robot
from config import CAMERA_WIDTH, CAMERA_HEIGHT

log = get_logger(__name__)

CAL_DRIVE_ROT = 2.0   # rotations forward
CAL_TURN_ROT  = 1.5   # rotations each direction


def _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners=None):
    """Grab frames until a valid pose is found. Returns (center_px, angle, corners)."""
    for _ in range(60):
        frame = stream.latest()
        if frame is None:
            time.sleep(0.05)
            continue
        frame = undistort_frame(frame, undist_maps)
        corners = detect_field(field_model, frame, last_corners)
        if corners is None:
            time.sleep(0.05)
            continue
        warped, homography = warp_field(frame, corners)
        h, w = warped.shape[:2]
        center, angle = get_true_robot_pose(aruco_detector, frame, homography, w, h)
        if center is not None:
            return center, angle, corners
        time.sleep(0.05)
    return None, None, last_corners


def main():
    setup_logging()
    log.info("GolfBot calibration tool")

    field_model    = load_field_model()
    aruco_detector = create_detector()
    stream         = open_stream()

    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))

    # Wait for C keypress to begin.
    log.info("Press C in the camera window to run calibration, any other key to quit.")
    while True:
        frame = stream.latest()
        if frame is None:
            time.sleep(0.05)
            continue
        frame = undistort_frame(frame, undist_maps)
        cv2.putText(frame, "C: run calibration   |   any key: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("GolfBot Calibration", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('c'):
            break
        elif key != 255:
            log.info("Calibration cancelled.")
            stream.stop()
            cv2.destroyAllWindows()
            return

    last_corners = None
    ok = True

    # --- Drive forward ---
    log.info("Driving forward %.1f rot...", CAL_DRIVE_ROT)
    start_px, _, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
    if start_px is None:
        log.warning("No pose before drive — aborting.")
        ok = False
    else:
        robot.drive(CAL_DRIVE_ROT)
        time.sleep(0.5)
        end_px, _, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
        if end_px is not None:
            measured = measure_pixels_per_rotation(start_px, end_px, CAL_DRIVE_ROT)
            calibration_pixels.update(measured)
            log.info("Drive → %.2f px/rot", calibration_pixels.ratio)
        else:
            log.warning("No pose after drive — skipping drive result.")
            ok = False

    # --- Turn left ---
    if ok:
        log.info("Turning left %.1f rot...", CAL_TURN_ROT)
        _, start_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
        if start_angle is None:
            log.warning("No pose before left turn — aborting.")
            ok = False
        else:
            robot.turn(CAL_TURN_ROT, "LEFT")
            time.sleep(0.5)
            _, end_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
            if end_angle is not None:
                measured = measure_degrees_per_rotation(start_angle, end_angle, CAL_TURN_ROT)
                calibration_angle_left.update(measured)
                log.info("Turn L → %.2f deg/rot", calibration_angle_left.ratio)
            else:
                log.warning("No pose after left turn — skipping.")

    # --- Turn right ---
    if ok:
        log.info("Turning right %.1f rot...", CAL_TURN_ROT)
        _, start_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
        if start_angle is None:
            log.warning("No pose before right turn — aborting.")
            ok = False
        else:
            robot.turn(CAL_TURN_ROT, "RIGHT")
            time.sleep(0.5)
            _, end_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
            if end_angle is not None:
                measured = measure_degrees_per_rotation(start_angle, end_angle, CAL_TURN_ROT)
                calibration_angle_right.update(measured)
                log.info("Turn R → %.2f deg/rot", calibration_angle_right.ratio)
            else:
                log.warning("No pose after right turn — skipping.")

    # Save to config.py so main.py picks them up.
    if ok:
        px, deg_l, deg_r = save_calibration_to_config()
        log.info("Saved to config — drive %.2f px/rot, L %.2f / R %.2f deg/rot", px, deg_l, deg_r)
    else:
        log.warning("Calibration incomplete — config.py not updated.")

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
