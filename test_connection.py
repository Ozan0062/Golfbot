"""
test_connection.py — quick sanity check.
Opens the camera and reads the gyro from the robot.
"""

import cv2
from vision.camera import open_camera, grab_frame, release
from ev3 import ev3_controller as robot


# Camera
print("Opening camera...")
cap = open_camera()
frame = grab_frame(cap)
print(f"Camera OK — frame size: {frame.shape[1]}x{frame.shape[0]}")
release(cap)

# Robot
print("Connecting to robot...")
angle = robot.get_angle()
print(f"Robot OK — gyro angle: {angle}°")
