"""
CAMERA MODULE

Handles access to the USB camera and returns frames for the vision system.
"""

import cv2


class Camera:
    def __init__(self, camera_index=0):
        """
        Initialize the camera.

        camera_index:
            0 = default webcam
            1+ = other connected cameras
        """
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise Exception("Could not open camera")

    def get_frame(self):
        """
        Capture a single frame from the camera.
        """
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("Failed to capture frame")

        return frame

    def release(self):
        """
        Release the camera when finished.
        """
        self.cap.release()


def test_camera():
    """
    Simple camera test that shows the live camera feed.
    Press ESC to exit.
    """

    cam = Camera()

    while True:
        frame = cam.get_frame()

        cv2.imshow("GolfBot Camera", frame)

        # ESC key to exit
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()