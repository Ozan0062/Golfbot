"""
test_drive_drift.py — Drive 300px forward and measure positional/angular drift.

Uses the live camera pipeline (undistort → field warp → ArUco) to record the
robot's pose before and after the drive, then prints the lateral drift, heading
drift, and actual distance vs requested distance.

Usage:
    python -m test.camera.test_drive_drift

The robot must be on the field with ArUco visible before this starts.
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.camera import open_stream
from vision.field import load_field_model, detect_corners, sort_corners, warp_field
from vision.aruco import create_detector
from vision.calibration import load_calibration, build_undistort_maps, remap
from vision.tracker import robot_px_to_cm, get_true_robot_pose
from controller.calibration_tracker import calibration_pixels
import controller.ev3_controller as robot
from config import CAMERA_WIDTH, CAMERA_HEIGHT


DRIVE_PX = 300
SETTLE_S = 1.5          # wait for robot to fully stop before reading pose
POSE_ATTEMPTS = 30      # max frames to try before giving up on ArUco


def get_pose(stream, field_model, aruco_detector, undist_maps, last_corners):
    """
    Grab frames until ArUco is detected.  Returns (center_px, angle, warped_corners)
    in warped coordinates (with height correction applied),
    or (None, None, last_corners) on failure.
    """
    for attempt in range(POSE_ATTEMPTS):
        frame = stream.latest()
        if frame is None:
            time.sleep(0.05)
            continue

        if undist_maps is not None:
            frame = remap(frame, *undist_maps)

        corners = detect_corners(field_model, frame)
        if len(corners) >= 4:
            last_corners = sort_corners(corners)

        if last_corners is None:
            time.sleep(0.05)
            continue

        warped, M = warp_field(frame, last_corners)
        h, w = warped.shape[:2]

        center_px, angle = get_true_robot_pose(
            aruco_detector, frame, M, w, h
        )
        if center_px is not None:
            return center_px, angle, last_corners

        time.sleep(0.05)

    return None, None, last_corners


def main():
    print("=== Drive Drift Test ===")
    print(f"Will drive {DRIVE_PX}px using calibration {calibration_pixels.ratio:.1f} px/rot\n")

    field_model    = load_field_model()
    aruco_detector = create_detector()
    stream         = open_stream()
    mtx, dist      = load_calibration()
    undist_maps    = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        print("Lens calibration loaded.")
    else:
        print("WARNING: No lens calibration — results will include distortion error.")

    last_corners = None

    # ── Wait for initial ArUco lock ─────────────────────────────────────
    print("Waiting for ArUco detection...")
    time.sleep(1)
    start_px, start_angle, last_corners = get_pose(
        stream, field_model, aruco_detector, undist_maps, last_corners
    )
    if start_px is None:
        print("ERROR: Could not detect ArUco marker. Is the robot on the field?")
        stream.stop()
        return

    start_cm = robot_px_to_cm(start_px, 640, 480)
    print(f"START  px=({start_px[0]:.1f}, {start_px[1]:.1f})  "
          f"cm=({start_cm[0]:.1f}, {start_cm[1]:.1f})  "
          f"heading={start_angle:.1f}°")

    # ── Drive ───────────────────────────────────────────────────────────
    rotations = DRIVE_PX / calibration_pixels.ratio
    print(f"\nDriving {DRIVE_PX}px → {rotations:.2f} motor rotations ...")
    robot.drive(rotations)

    print(f"Settling {SETTLE_S}s ...")
    time.sleep(SETTLE_S)

    # ── Read end pose ───────────────────────────────────────────────────
    end_px, end_angle, last_corners = get_pose(
        stream, field_model, aruco_detector, undist_maps, last_corners
    )
    if end_px is None:
        print("ERROR: Lost ArUco after drive. Cannot measure drift.")
        stream.stop()
        return

    end_cm = robot_px_to_cm(end_px, 640, 480)
    print(f"END    px=({end_px[0]:.1f}, {end_px[1]:.1f})  "
          f"cm=({end_cm[0]:.1f}, {end_cm[1]:.1f})  "
          f"heading={end_angle:.1f}°")

    # ── Compute drift ───────────────────────────────────────────────────
    dx = end_px[0] - start_px[0]
    dy = end_px[1] - start_px[1]
    actual_dist_px = math.hypot(dx, dy)

    # Heading of the travel vector
    travel_angle = math.degrees(math.atan2(dy, dx))

    # Lateral drift = component perpendicular to the intended heading
    heading_rad = math.radians(start_angle)
    forward_component = dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
    lateral_component = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)

    heading_drift = ((end_angle - start_angle) + 180) % 360 - 180

    print(f"\n{'─' * 50}")
    print(f"RESULTS")
    print(f"{'─' * 50}")
    print(f"  Requested distance:  {DRIVE_PX} px")
    print(f"  Actual distance:     {actual_dist_px:.1f} px")
    print(f"  Distance error:      {actual_dist_px - DRIVE_PX:+.1f} px  "
          f"({(actual_dist_px - DRIVE_PX) / DRIVE_PX * 100:+.1f}%)")
    print(f"  Forward component:   {forward_component:.1f} px")
    print(f"  Lateral drift:       {lateral_component:+.1f} px  "
          f"({'left' if lateral_component < 0 else 'right'})")
    print(f"  Heading before:      {start_angle:.1f}°")
    print(f"  Heading after:       {end_angle:.1f}°")
    print(f"  Heading drift:       {heading_drift:+.1f}°")
    print(f"  Travel vector angle: {travel_angle:.1f}°")
    print(f"  Calibration used:    {calibration_pixels.ratio:.1f} px/rot")
    print(f"  Motor rotations:     {rotations:.2f}")
    print(f"{'─' * 50}")

    stream.stop()


if __name__ == "__main__":
    main()
