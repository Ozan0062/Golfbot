#capture frames from overhead USB camera
#
# Run standalone to test your camera: python -m vision.camera

import cv2
import sys
import time
import os
from datetime import datetime
sys.path.append(".")
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


def open_camera(index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    """Open camera and return the VideoCapture object."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {index}")

    return cap


def grab_frame(cap):
    """Read a single frame. Returns the frame or raises on failure."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab frame from camera")
    return frame


def release(cap):
    """Clean up camera and any OpenCV windows."""
    cap.release()
    cv2.destroyAllWindows()


# ── Standalone test ─────────────────────────────────
# Auto-saves 72 pictures (one every 5 seconds) to images-robot/, then exits.
# Press ESC to quit early.
if __name__ == "__main__":
    TOTAL_IMAGES = 50
    INTERVAL = 5.0

    cap = open_camera()
    os.makedirs("images-robot", exist_ok=True)
    print(f"Camera opened ({CAMERA_WIDTH}x{CAMERA_HEIGHT}). Taking {TOTAL_IMAGES} images every {INTERVAL}s. Press ESC to quit early.")

    # Continue count from existing images
    existing = [f for f in os.listdir("images-robot") if f.endswith(".jpg") and f[:-4].isdigit()]
    existing_count = max((int(f[:-4]) for f in existing), default=0)
    count = existing_count
    if count:
        print(f"Resuming from image {count + 1} (taking {TOTAL_IMAGES} more).")
    last_save = time.time() - INTERVAL  # trigger immediately on first frame

    while count < existing_count + TOTAL_IMAGES:
        frame = grab_frame(cap)
        cv2.imshow("Camera Test", frame)

        now = time.time()
        if now - last_save >= INTERVAL:
            count += 1
            filename = os.path.join("images-robot", f"{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{count}/{TOTAL_IMAGES}] Saved {filename}")
            last_save = now

        key = cv2.waitKey(1) & 0xFF
        if key == 27:       # ESC
            print("Aborted early.")
            break

    print(f"Done. {count} images saved to images-robot/")
    release(cap)