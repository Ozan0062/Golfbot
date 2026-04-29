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
    """Open camera, auto-detecting index if the configured one fails."""
    for i in ([index] + [x for x in range(5) if x != index]):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, _ = cap.read()
        if cap.isOpened() and ret:
            if i != index:
                print(f"Camera not found at index {index}, using index {i} instead.")
            return cap
        cap.release()

    raise RuntimeError("Could not find any working camera (tried indices 0-4)")


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
# Press SPACE to capture an image, ESC to quit.
# Saves to images-robot/, continuing the count from existing images.
if __name__ == "__main__":
    TOTAL_IMAGES = 100

    cap = open_camera()
    os.makedirs("images-robot", exist_ok=True)

    # Continue count from existing images
    existing = [f for f in os.listdir("images-robot") if f.endswith(".jpg") and f[:-4].isdigit()]
    existing_count = max((int(f[:-4]) for f in existing), default=0)
    count = existing_count
    target = existing_count + TOTAL_IMAGES
    if count:
        print(f"Resuming from image {count + 1}. {TOTAL_IMAGES} more to go.")
    print(f"Camera opened ({CAMERA_WIDTH}x{CAMERA_HEIGHT}). Press SPACE to capture, ESC to quit.")

    while count < target:
        frame = grab_frame(cap)
        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC
            print("Aborted early.")
            break
        elif key == ord(" "):   # SPACE
            count += 1
            filename = os.path.join("images-robot", f"{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{count}/{target}] Saved {filename}")

    print(f"Done. {count} images saved to images-robot/")
    release(cap)