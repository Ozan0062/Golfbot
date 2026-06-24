"""
zone_calibrate.py — zone-based ongoing calibration tool.

Drives the robot through a systematic pattern across all four field quadrants,
measuring drive and turn ratios in each zone.  The longer the script runs, the
more accurate the calibration becomes.

Usage:
    python zone_calibrate.py

Press C in the camera window to start an iteration.
Press ESC at any time to save and quit.

The calibration is saved to zone_calibration.json after every complete
iteration and on exit.
"""

import math
import time
import sys

import cv2

from golfbot_logger import setup_logging, get_logger
from vision.camera          import open_stream
from vision.field           import load_field_model, warp_field, detect_field
from vision.tracker         import get_true_robot_pose
from vision.aruco           import create_detector
from vision.lens_calibration import load_calibration, build_undistort_maps, undistort_frame
from controller.drive_calibration import (
    measure_pixels_per_rotation, measure_degrees_per_rotation,
)
from controller.zone_calibration_tracker import zone_tracker, get_zone
import controller.ev3_controller as robot
from config import CAMERA_WIDTH, CAMERA_HEIGHT, WARPED_WIDTH, WARPED_HEIGHT, ZONE_CENTER_PX

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Calibration movement parameters
# --------------------------------------------------------------------------
CAL_DRIVE_ROT = 1.0    # rotations per drive test (shorter to avoid walls)
CAL_TURN_ROT  = 1.5    # rotations per turn test (each direction)
SETTLE_WAIT   = 0.6    # seconds to wait after a motor command before measuring
POSE_ATTEMPTS = 80     # max frames to try for a valid pose

# --------------------------------------------------------------------------
# Zone target points — one per zone, well inside each quadrant so movements
# stay within the same zone.
# --------------------------------------------------------------------------
_MARGIN = 150  # px from the zone border to keep all movement within one zone, safely away from walls
_CX, _CY = ZONE_CENTER_PX

ZONE_TARGETS = {
    0: (_MARGIN,                   _MARGIN),                    # TL
    1: (WARPED_WIDTH - _MARGIN,    _MARGIN),                    # TR
    2: (_MARGIN,                   WARPED_HEIGHT - _MARGIN),    # BL
    3: (WARPED_WIDTH - _MARGIN,    WARPED_HEIGHT - _MARGIN),    # BR
}

# Order of zones in the calibration route (a loop through all 4).
ZONE_ORDER = [0, 1, 3, 2]  # TL → TR → BR → BL → back to TL


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners=None):
    """Grab frames until a valid pose is found.  Returns (center_px, angle, corners)."""
    for _ in range(POSE_ATTEMPTS):
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


def _show_overlay(stream, undist_maps, field_model, last_corners, text_lines, wait_ms=1):
    """Show the camera feed with status text; return True if ESC was pressed."""
    frame = stream.latest()
    if frame is None:
        return False
    frame = undistort_frame(frame, undist_maps)
    corners = detect_field(field_model, frame, last_corners)
    if corners is not None:
        warped, _ = warp_field(frame, corners)
        display = warped
    else:
        display = frame

    for i, line in enumerate(text_lines):
        cv2.putText(display, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("Zone Calibration", display)
    return (cv2.waitKey(wait_ms) & 0xFF) == 27


def _navigate_to(target_px, stream, undist_maps, field_model, aruco_detector,
                 last_corners, arrive_radius=40):
    """
    Simple bang-bang controller to drive the robot to a target point.
    Returns the last_corners (for field detection continuity) or None on ESC.
    """
    from controller.navigation import angle_to_target, angle_error, px_to_cm
    from controller.calibration_tracker import calibration_pixels, calibration_angle_left, calibration_angle_right

    for step in range(200):  # safety limit
        center, angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
        if center is None:
            log.warning("Lost pose during navigation — retrying")
            time.sleep(0.2)
            continue

        dist = math.hypot(center[0] - target_px[0], center[1] - target_px[1])
        if dist <= arrive_radius:
            log.info("Arrived at (%.0f, %.0f) — dist %.0f px", target_px[0], target_px[1], dist)
            return last_corners

        # Heading error
        robot_cm  = px_to_cm(center)
        target_cm = px_to_cm(target_px)
        desired   = angle_to_target(robot_cm, target_cm)
        err       = angle_error(angle, desired)

        if abs(err) > 12:
            # Turn to face the target (wider tolerance so it doesn't micro-adjust)
            rots = abs(err) / (calibration_angle_left.ratio if err < 0 else calibration_angle_right.ratio)
            rots = max(rots, 0.15)
            direction = "LEFT" if err < 0 else "RIGHT"
            robot.turn(rots, direction)
            time.sleep(SETTLE_WAIT)
        else:
            # Drive toward target (much faster/larger chunks)
            drive_px = min(dist, 250)  
            rots = drive_px / calibration_pixels.ratio
            rots = max(rots, 0.15)
            robot.drive(rots)
            time.sleep(SETTLE_WAIT)

        if _show_overlay(stream, undist_maps, field_model, last_corners,
                         [f"Navigating to zone target ({target_px[0]:.0f}, {target_px[1]:.0f})",
                          f"dist={dist:.0f}px  heading_err={err:.1f} deg"]):
            return None  # ESC

    log.warning("Navigation timed out after 200 steps")
    return last_corners


def _calibrate_in_zone(zone_id, stream, undist_maps, field_model, aruco_detector, last_corners):
    """
    Run one drive + turn-left + turn-right calibration sequence in the current zone.
    Returns updated last_corners, or None if aborted.
    """
    zone_name = {0: "TL", 1: "TR", 2: "BL", 3: "BR"}[zone_id]
    log.info("=== Calibrating zone %d (%s) ===", zone_id, zone_name)

    # --- Drive test ---
    log.info("Zone %d: Drive forward %.1f rot...", zone_id, CAL_DRIVE_ROT)
    start_px, start_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
    if start_px is None:
        log.warning("Zone %d: No pose before drive — skipping drive", zone_id)
    else:
        robot.drive(CAL_DRIVE_ROT)
        time.sleep(SETTLE_WAIT)
        end_px, _, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
        if end_px is not None:
            start_zone = get_zone(start_px, zone_tracker.center_px)
            end_zone   = get_zone(end_px, zone_tracker.center_px)
            if start_zone == end_zone == zone_id:
                measured = measure_pixels_per_rotation(start_px, end_px, CAL_DRIVE_ROT)
                zone_tracker.update_drive(zone_id, measured)
                log.info("Zone %d drive: %.2f px/rot", zone_id, zone_tracker.zones[zone_id].px_per_rot)
            else:
                log.warning("Zone %d: Drive crossed zone boundary (%s→%s) — skipping",
                            zone_id, start_zone, end_zone)
        else:
            log.warning("Zone %d: No pose after drive — skipping drive measurement", zone_id)

    if _show_overlay(stream, undist_maps, field_model, last_corners,
                     [f"Zone {zone_id} ({zone_name}): drive done"]):
        return None

    # --- Turn left test ---
    log.info("Zone %d: Turn left %.1f rot...", zone_id, CAL_TURN_ROT)
    _, start_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
    if start_angle is None:
        log.warning("Zone %d: No pose before left turn — skipping", zone_id)
    else:
        robot.turn(CAL_TURN_ROT, "LEFT")
        time.sleep(SETTLE_WAIT)
        _, end_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
        if end_angle is not None:
            measured = measure_degrees_per_rotation(start_angle, end_angle, CAL_TURN_ROT)
            zone_tracker.update_turn(zone_id, measured, "LEFT")
            log.info("Zone %d turn L: %.2f deg/rot", zone_id, zone_tracker.zones[zone_id].deg_per_rot_left)
        else:
            log.warning("Zone %d: No pose after left turn — skipping", zone_id)

    if _show_overlay(stream, undist_maps, field_model, last_corners,
                     [f"Zone {zone_id} ({zone_name}): left turn done"]):
        return None

    # --- Turn right test ---
    log.info("Zone %d: Turn right %.1f rot...", zone_id, CAL_TURN_ROT)
    _, start_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
    if start_angle is None:
        log.warning("Zone %d: No pose before right turn — skipping", zone_id)
    else:
        robot.turn(CAL_TURN_ROT, "RIGHT")
        time.sleep(SETTLE_WAIT)
        _, end_angle, last_corners = _get_pose(stream, undist_maps, field_model, aruco_detector, last_corners)
        if end_angle is not None:
            measured = measure_degrees_per_rotation(start_angle, end_angle, CAL_TURN_ROT)
            zone_tracker.update_turn(zone_id, measured, "RIGHT")
            log.info("Zone %d turn R: %.2f deg/rot", zone_id, zone_tracker.zones[zone_id].deg_per_rot_right)
        else:
            log.warning("Zone %d: No pose after right turn — skipping", zone_id)

    if _show_overlay(stream, undist_maps, field_model, last_corners,
                     [f"Zone {zone_id} ({zone_name}): calibration complete"]):
        return None

    return last_corners


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    setup_logging()
    log.info("Zone Calibration Tool")
    log.info("Field centre: (%d, %d)  |  4 zones", _CX, _CY)

    field_model    = load_field_model()
    aruco_detector = create_detector()
    stream         = open_stream()

    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))

    # Show current zone state
    print("\n" + zone_tracker.summary_table() + "\n")

    log.info("Press C in the camera window to start calibration, ESC to quit.")
    while True:
        if _show_overlay(stream, undist_maps, field_model, None,
                         ["Press C to start zone calibration", "ESC to quit"],
                         wait_ms=50):
            _save_and_exit(stream)
            return

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif key == 27:
            _save_and_exit(stream)
            return

    iteration = 0
    last_corners = None

    try:
        while True:
            iteration += 1
            log.info("══════════════════════════════════════════════════")
            log.info("  ITERATION %d", iteration)
            log.info("══════════════════════════════════════════════════")

            aborted = False
            for zone_id in ZONE_ORDER:
                target = ZONE_TARGETS[zone_id]
                zone_name = {0: "TL", 1: "TR", 2: "BL", 3: "BR"}[zone_id]

                log.info("--- Navigating to zone %d (%s) at (%.0f, %.0f) ---",
                         zone_id, zone_name, target[0], target[1])

                last_corners = _navigate_to(
                    target, stream, undist_maps, field_model,
                    aruco_detector, last_corners,
                )
                if last_corners is None:
                    aborted = True
                    break

                last_corners = _calibrate_in_zone(
                    zone_id, stream, undist_maps, field_model,
                    aruco_detector, last_corners,
                )
                if last_corners is None:
                    aborted = True
                    break

            if aborted:
                log.info("Iteration %d aborted by user", iteration)
                break

            # Save after each complete iteration
            zone_tracker.save()

            # Print summary
            print(f"\n{'═' * 60}")
            print(f"  Iteration {iteration} complete")
            print(f"{'═' * 60}")
            print(zone_tracker.summary_table())
            print(f"\nTotal drive samples: {sum(zd.samples_drive for zd in zone_tracker.zones.values())}")
            print(f"Total turn samples:  {sum(zd.samples_turn_left + zd.samples_turn_right for zd in zone_tracker.zones.values())}")
            print(f"{'═' * 60}\n")

            log.info("Press C for another iteration, ESC to save and quit.")

            # Wait for user input between iterations
            while True:
                if _show_overlay(stream, undist_maps, field_model, last_corners,
                                 [f"Iteration {iteration} done — C: next iteration, ESC: quit"],
                                 wait_ms=100):
                    aborted = True
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
                elif key == 27:
                    aborted = True
                    break

            if aborted:
                break

    finally:
        _save_and_exit(stream)


def _save_and_exit(stream):
    """Save calibration and clean up."""
    zone_tracker.save()
    stream.stop()
    cv2.destroyAllWindows()

    print("\n" + "═" * 60)
    print("  Final Zone Calibration Results")
    print("═" * 60)
    print(zone_tracker.summary_table())
    print("═" * 60 + "\n")
    log.info("Zone calibration saved — exiting.")


if __name__ == "__main__":
    main()
