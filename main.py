#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile


# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.


# Create your objects here.
ev3 = EV3Brick()


# Write your program here.
ev3.speaker.beep()


"""
MAIN PROGRAM

This file runs the full GolfBot pipeline.

Pipeline:

Camera
  ↓
Image Recognition
  ↓
Navigation
  ↓
Send command to EV3 robot
"""

# Example imports (to be implemented later)
# from ImageRecognitionModule.camera import get_frame
# from ImageRecognitionModule.detect_ball import detect_ball
# from Navigation.navigation import decide_movement
# from ev3_controller import send_command


def main():
    print("GolfBot system starting...")

    while True:

        # Step 1: Get camera frame
        # frame = get_frame()

        # Step 2: Detect objects
        # ball_x, ball_y = detect_ball(frame)

        # Step 3: Decide movement
        # move = decide_movement(ball_x)

        # Step 4: Send command to robot
        # send_command(move)

        print("Pipeline running...")


if __name__ == "__main__":
    main()