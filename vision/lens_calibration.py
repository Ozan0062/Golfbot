"""
lens_calibration.py - camera lens-distortion calibration.

Workflow (run the tool from the repo root):
    1. Print a 9x6 inner-corner checkerboard.
    2. python -m scripts.lens_calibration_tool capture    - capture 10-15 images
    3. python -m scripts.lens_calibration_tool calibrate  - compute + save calibration
    4. Done - main.py loads vision/calibration_data.npz and undistorts automatically.

This module holds the calibration functions; the interactive CLI lives in
scripts/lens_calibration_tool.py.
"""

import os
import glob
import sys
sys.path.append(".")

import cv2
import numpy as np

CALIBRATION_DIR = "vision/calibration_images"
CALIBRATION_FILE = "vision/calibration_data.npz"

# Checkerboard dimensions (inner corners, not squares).
# Standard printable checkerboard: 10x7 squares = 9x6 inner corners.
CHECKERBOARD = (9, 6)


def capture_checkerboard_images():
    """Open camera and capture checkerboard images with SPACE. ESC to quit."""
    from vision.camera import open_camera, grab_frame, release

    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    cap = open_camera()
    count = len(glob.glob(f"{CALIBRATION_DIR}/*.jpg"))
    print(f"Hold checkerboard in front of camera. Press SPACE to capture, ESC when done.")
    print(f"Capture from different angles and distances. Aim for 10-15 images.")
    if count:
        print(f"Already have {count} images, continuing from there.")

    while True:
        frame = grab_frame(cap)
        display = frame.copy()

        # Try to find checkerboard in real-time for visual feedback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
            cv2.putText(display, "Checkerboard found! Press SPACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No checkerboard detected...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Images: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(" ") and found:
            count += 1
            filename = f"{CALIBRATION_DIR}/calib_{count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  Saved {filename} ({count} total)")

    release(cap)
    print(f"\nDone. {count} images saved to {CALIBRATION_DIR}/")


def calibrate():
    """Compute calibration from saved checkerboard images."""
    images = sorted(glob.glob(f"{CALIBRATION_DIR}/*.jpg"))
    if len(images) < 5:
        print(f"Need at least 5 images, found {len(images)}. Run 'capture' first.")
        return

    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image
    img_size = None

    print(f"Processing {len(images)} images...")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners)
            print(f"  {fname}: found")
        else:
            print(f"  {fname}: no checkerboard found, skipping")

    if len(obj_points) < 5:
        print(f"Only {len(obj_points)} usable images. Need at least 5.")
        return

    print(f"\nCalibrating with {len(obj_points)} images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"  Reprojection error: {ret:.4f} (lower is better, <0.5 is good)")
    print(f"  Camera matrix:\n{mtx}")
    print(f"  Distortion coefficients: {dist.ravel()}")

    np.savez(CALIBRATION_FILE, mtx=mtx, dist=dist)
    print(f"\nCalibration saved to {CALIBRATION_FILE}")


def load_calibration():
    """Load calibration data. Returns (mtx, dist) or (None, None) if not calibrated."""
    if not os.path.exists(CALIBRATION_FILE):
        return None, None
    data = np.load(CALIBRATION_FILE)
    return data["mtx"], data["dist"]


def undistort(frame, mtx, dist):
    """Apply lens distortion correction to a frame (slow - recomputes map every call)."""
    return cv2.undistort(frame, mtx, dist)


def build_undistort_maps(mtx, dist, size):
    """
    Build remap lookup tables once at startup.
    size = (width, height).  Returns (map1, map2) for cv2.remap().
    """
    w, h = size
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2)


def remap(frame, map1, map2):
    """Fast undistortion using precomputed maps."""
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def undistort_frame(frame, undist_maps):
    """Apply lens correction if calibration data is available."""
    if undist_maps is not None:
        return remap(frame, *undist_maps)
    return frame
