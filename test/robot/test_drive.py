from mov_control import drive_forward, turn_left, turn_right, stop
import time

print("Forward 2s...")
drive_forward()
time.sleep(2)
stop()
time.sleep(0.5)

print("Left 1s...")
turn_left()
time.sleep(1)
stop()
time.sleep(0.5)

print("Right 1s...")
turn_right()
time.sleep(1)
stop()

print("Done.")
