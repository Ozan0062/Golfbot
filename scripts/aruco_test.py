"""
aruco_test.py — live ArUco robot-pose detection (and regenerate the marker).

Run:  python -m scripts.aruco_test
Press ESC to quit.
"""

import cv2

from vision.camera import open_camera, grab_frame, release
from vision.aruco import create_detector, detect_robot, draw_robot, generate_marker


def main():
    # Generate printable marker
    generate_marker()

    cap = open_camera()
    detector = create_detector()
    print("ArUco detection running. Press ESC to quit.")

    while True:
        frame = grab_frame(cap)
        center, angle = detect_robot(detector, frame)

        if center:
            frame = draw_robot(frame, center, angle)
            cv2.putText(frame, f"pos=({center[0]:.0f},{center[1]:.0f}) angle={angle:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No marker found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ArUco Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    release(cap)


if __name__ == "__main__":
    main()
