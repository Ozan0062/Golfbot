#capture frames from overhead USB camera
#
# Run standalone to test your camera: python -m vision.camera

import cv2
import sys
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
# Shows live feed. Press ESC to quit, 's' to save a snapshot.
if __name__ == "__main__":
    cap = open_camera()
    print(f"Camera opened ({CAMERA_WIDTH}x{CAMERA_HEIGHT}). Press ESC to quit, 's' to save snapshot.")

    while True:
        frame = grab_frame(cap)
        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:       # ESC
            break
        elif key == ord("s"):
            cv2.imwrite("snapshot.jpg", frame)
            print("Saved snapshot.jpg")

    release(cap)