from robot.mov_control import drive_forward, drive_backward, turn_left, turn_right, stop
from time import sleep

print("Driving forward")
drive_forward()

sleep(2)

print("Driving backwards")
drive_backward()

sleep(2)

print("Turn right")
turn_right()

sleep(2)

print("Turn left")
turn_left()