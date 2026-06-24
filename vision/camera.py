"""
camera.py — capture frames from the overhead USB camera.

Standalone capture tool (save training images):
    python -m scripts.camera_capture
"""

import sys
import threading
sys.path.append(".")

import cv2

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


def open_camera(index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    """Open camera, auto-detecting index if the configured one fails."""
    for i in ([index] + [x for x in range(5) if x != index]):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, _ = cap.read()
        if cap.isOpened() and ret:
            if i != index:
                print(f"Camera not found at index {index}, using index {i} instead.")
            return cap
        cap.release()

    raise RuntimeError("Could not find any working camera (tried indices 0-4)")


class CameraStream:
    """
    Background thread that continuously drains the camera buffer.

    The thread calls cap.read() in a tight loop and stores the result.
    Because it never stops reading, the buffer never builds up — even
    during long blocking EV3 moves.

    Call latest() to get the most recent frame.  No flushing needed.
    """

    def __init__(self, cap):
        self._cap     = cap
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def latest(self):
        """Return a copy of the most recently captured frame, or None if not ready yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Stop the background thread and release the camera."""
        self._running = False
        self._cap.release()
        cv2.destroyAllWindows()


def open_stream(index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT) -> CameraStream:
    """Open camera and start background capture thread. Use in main loop."""
    cap = open_camera(index, width, height)
    return CameraStream(cap)


def grab_frame(cap):
    """
    Grab a single frame from a raw VideoCapture.
    Used by standalone test scripts that don't need the stream thread.
    """
    for _ in range(2):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab frame from camera")
    return frame


def release(cap):
    """Clean up a raw VideoCapture and any OpenCV windows."""
    cap.release()
    cv2.destroyAllWindows()
