import sys, os
sys.path.insert(0, os.path.expanduser("~/"))

from mov_control import drive_forward, turn_right

FORWARD_ROTATIONS = 6.0
TURN_ROTATIONS    = 3.0   # nudge until corners are 90 degrees

for _ in range(4):
    drive_forward(FORWARD_ROTATIONS)
    turn_right(TURN_ROTATIONS)

print("Done.")
