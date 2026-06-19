"""
lens_calibration_tool.py — CLI for camera lens calibration.

Run from the repo root:
    python -m scripts.lens_calibration_tool capture    — take checkerboard photos
    python -m scripts.lens_calibration_tool calibrate  — compute calibration
    python -m scripts.lens_calibration_tool test       — show undistorted live feed
"""

import sys
import cv2
import numpy as np

from vision.lens_calibration import (
    capture_checkerboard_images, calibrate, load_calibration, undistort,
)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("capture", "calibrate", "test"):
        print("Usage:")
        print("  python -m scripts.lens_calibration_tool capture    — take checkerboard photos")
        print("  python -m scripts.lens_calibration_tool calibrate  — compute calibration")
        print("  python -m scripts.lens_calibration_tool test       — show undistorted live feed")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "capture":
        capture_checkerboard_images()

    elif cmd == "calibrate":
        calibrate()

    elif cmd == "test":
        from vision.camera import open_camera, grab_frame, release
        mtx, dist = load_calibration()
        if mtx is None:
            print("No calibration found. Run 'capture' then 'calibrate' first.")
            sys.exit(1)

        cap = open_camera()
        print("Showing raw (left) vs undistorted (right). Press ESC to quit.")

        while True:
            frame = grab_frame(cap)
            corrected = undistort(frame, mtx, dist)

            combined = np.hstack([frame, corrected])
            cv2.putText(combined, "Raw", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(combined, "Undistorted", (frame.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration Test", combined)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        release(cap)


if __name__ == "__main__":
    main()
