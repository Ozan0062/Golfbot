"""
test_step_main.py — Manual step-through of the full GolfBot main loop.

Runs the complete pipeline (camera → field warp → ArUco → YOLO → world →
state machine) but pauses after every state machine tick.

Controls:
    SPACE  — advance one tick (runs controller.update and sends the command)
    ESC    — quit

Usage:
    python -m test.camera.test_step_main
"""

import sys
import os
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.camera      import open_stream
from vision.field       import load_field_model, warp_field, detect_field
from vision.detector    import load_object_model, detect_objects, draw_debug_overlay
from vision.tracker     import get_true_robot_pose, filter_detections_near_robot, build_world_dict
from vision.aruco       import create_detector
from vision.calibration import load_calibration, build_undistort_maps, undistort_frame

from controller.state_machine import GolfBotController
from config import CAMERA_WIDTH, CAMERA_HEIGHT


def main():
    print("=== Step-Through Main Test ===")
    print("SPACE = advance one tick    ESC = quit\n")

    field_model    = load_field_model()
    object_model   = load_object_model()
    aruco_detector = create_detector()
    stream         = open_stream()

    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        print("Lens calibration loaded.")
    else:
        print("WARNING: No lens calibration.")

    controller   = GolfBotController()
    last_corners = None
    step         = 0

    while True:
        # ── Grab and undistort frame ─────────────────────────────────────
        frame = stream.latest()
        if frame is None:
            time.sleep(0.05)
            continue

        frame = undistort_frame(frame, undist_maps)

        # ── Detect field ─────────────────────────────────────────────────
        last_corners = detect_field(field_model, frame, last_corners)
        if last_corners is None:
            cv2.imshow("GolfBot Step", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            continue

        warped, homography = warp_field(frame, last_corners)
        h, w = warped.shape[:2]

        # ── Detect robot and balls ────────────────────────────────────────
        robot_center, robot_angle = get_true_robot_pose(
            aruco_detector, frame, homography, w, h
        )
        detections = detect_objects(object_model, warped)
        detections = filter_detections_near_robot(detections, robot_center)
        world      = build_world_dict(detections, robot_center, robot_angle, w, h)

        # ── Draw current state (before tick) — label shows "PAUSED" ──────
        state_label = f"[{step}] {controller.state.name} — SPACE to step"
        debug = draw_debug_overlay(
            warped, detections, robot_center, robot_angle,
            state_label, "", controller._locked_target,
        )
        cv2.imshow("GolfBot Step", debug)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:          # ESC
            break
        if key != 32:          # not SPACE — just refresh the view
            continue

        # ── SPACE pressed: run one tick ───────────────────────────────────
        step += 1
        command = controller.update(world)
        print(f"\n[STEP {step}]  state={controller.state.name}  cmd={command.name}")

        # ── Redraw with the command that was just issued ──────────────────
        debug = draw_debug_overlay(
            warped, detections, robot_center, robot_angle,
            controller.state.name, command.name, controller._locked_target,
        )
        cv2.imshow("GolfBot Step", debug)
        cv2.waitKey(1)

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
