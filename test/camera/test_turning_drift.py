"""
test_turning_drift.py — Turn in place and calculate the QR→centre-of-rotation offset.

Turns the robot and observes how the QR marker position shifts relative to
the heading change.  If the QR marker is exactly at the centre of rotation,
it won't move at all during a turn.  Any positional drift reveals the offset
between the QR marker and the true pivot point.

Given start/end QR positions and headings, the offset (ox, oy) in the
robot's local frame satisfies:

    QR_end - QR_start = (R(θ_end) - R(θ_start)) · [ox, oy]

Solving this 2×2 system yields the offset.

Usage:
    python -m test.camera.test_turning_drift
    python -m test.camera.test_turning_drift --degrees 90
"""

import sys
import os
import time
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.camera import open_stream
from vision.field import load_field_model, detect_corners, sort_corners, warp_field
from vision.aruco import create_detector
from vision.lens_calibration import load_calibration, build_undistort_maps, remap
from vision.tracker import robot_px_to_cm, detect_robot_pose_in_warped_coords
from controller.calibration_tracker import calibration_angle_left
import controller.ev3_controller as robot
from config import CAMERA_WIDTH, CAMERA_HEIGHT


SETTLE_S      = 5.0
POSE_ATTEMPTS = 30


def get_pose(stream, field_model, aruco_detector, undist_maps, last_corners):
    """
    Grab frames until ArUco is detected.
    Returns (center_px, angle_deg, last_corners) in warped coords.
    """
    for _ in range(POSE_ATTEMPTS):
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

        center_px, _forward, angle = detect_robot_pose_in_warped_coords(
            aruco_detector, frame, M
        )
        if center_px is not None:
            return center_px, angle, last_corners

        time.sleep(0.05)

    return None, None, last_corners


def calc_offset(start_px, start_angle, end_px, end_angle):
    """
    Solve for (ox, oy) in the robot's local frame given QR drift and heading change.

    The QR marker sits at C + R(θ)·[ox,oy] where C is the centre of rotation.
    After turning:  QR_end - QR_start = (R(θ_end) - R(θ_start)) · [ox, oy]

    Returns (ox, oy) in pixels, or None if heading didn't change enough.
    """
    θ0 = math.radians(start_angle)
    θ1 = math.radians(end_angle)

    a = math.cos(θ1) - math.cos(θ0)
    b = math.sin(θ1) - math.sin(θ0)
    det = a * a + b * b

    if det < 1e-6:
        return None  # heading barely changed, can't solve

    dx = end_px[0] - start_px[0]
    dy = end_px[1] - start_px[1]

    ox = (a * dx + b * dy) / det
    oy = (a * dy - b * dx) / det
    return (ox, oy)


def main():
    parser = argparse.ArgumentParser(description="Calculate QR→centre offset from turning drift")
    parser.add_argument("--degrees", type=float, default=200,
                        help="Degrees to turn (default 360)")
    args = parser.parse_args()

    turn_deg = args.degrees

    print("=== QR Offset Calculator ===")
    print(f"Calibration: {calibration_angle_left.ratio:.1f} deg/rot")
    print(f"Turn amount: {turn_deg}°\n")

    field_model    = load_field_model()
    aruco_detector = create_detector()
    stream         = open_stream()
    mtx, dist      = load_calibration()
    undist_maps    = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        print("Lens calibration loaded.")
    else:
        print("WARNING: No lens calibration — results include distortion error.")

    last_corners = None
    rotations = turn_deg / calibration_angle_left.ratio

    # ── Read start pose ───────────────────────────────────────────────
    print("Waiting for ArUco detection...")
    time.sleep(1)
    start_px, start_angle, last_corners = get_pose(
        stream, field_model, aruco_detector, undist_maps, last_corners
    )
    if start_px is None:
        print("ERROR: Could not detect ArUco. Is the robot on the field?")
        stream.stop()
        return

    print(f"START  px=({start_px[0]:.1f}, {start_px[1]:.1f})  heading={start_angle:.1f}°")

    results = []

    for direction in ["LEFT", "RIGHT"]:
        print(f"\n{'═' * 50}")
        print(f"  Turning {turn_deg}° {direction} ({rotations:.2f} motor rotations)")
        print(f"{'═' * 50}")
        robot.turn(rotations, direction)

        print(f"Settling {SETTLE_S}s ...")
        time.sleep(SETTLE_S)

        end_px, end_angle, last_corners = get_pose(
            stream, field_model, aruco_detector, undist_maps, last_corners
        )
        if end_px is None:
            print(f"ERROR: Lost ArUco after {direction} turn.")
            stream.stop()
            return

        heading_drift = ((end_angle - start_angle) + 180) % 360 - 180
        qr_drift = math.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1])

        print(f"END    px=({end_px[0]:.1f}, {end_px[1]:.1f})  heading={end_angle:.1f}°")
        print(f"  QR drift:      {qr_drift:.1f} px")
        print(f"  Heading drift: {heading_drift:+.1f}°")

        offset = calc_offset(start_px, start_angle, end_px, end_angle)
        if offset is None:
            print("  Could not calculate offset (heading barely changed).")
        else:
            print(f"  Calculated offset: ox={offset[0]:+.1f}  oy={offset[1]:+.1f} px")
            print(f"  Offset magnitude:  {math.hypot(*offset):.1f} px")
            results.append({"direction": direction, "ox": offset[0], "oy": offset[1]})

        # Use end as next start
        start_px, start_angle = end_px, end_angle

    # ── Summary ───────────────────────────────────────────────────────
    if len(results) == 2:
        avg_ox = (results[0]["ox"] + results[1]["ox"]) / 2
        avg_oy = (results[0]["oy"] + results[1]["oy"]) / 2
        print(f"\n{'═' * 50}")
        print("  OFFSET SUMMARY")
        print(f"{'═' * 50}")
        print(f"  LEFT  turn:  ox={results[0]['ox']:+.1f}  oy={results[0]['oy']:+.1f}")
        print(f"  RIGHT turn:  ox={results[1]['ox']:+.1f}  oy={results[1]['oy']:+.1f}")
        print(f"  Average:     ox={avg_ox:+.1f}  oy={avg_oy:+.1f}")
        print(f"  Magnitude:   {math.hypot(avg_ox, avg_oy):.1f} px")

        diff = math.hypot(results[0]["ox"] - results[1]["ox"],
                          results[0]["oy"] - results[1]["oy"])
        if diff < 10:
            print("  ✓  Left/right agree — offset is reliable.")
        else:
            print(f"  ⚠  Left/right disagree by {diff:.1f} px — mechanical asymmetry?")

    stream.stop()


if __name__ == "__main__":
    main()
