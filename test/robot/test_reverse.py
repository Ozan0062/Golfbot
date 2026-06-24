#!/usr/bin/env python3
"""Reverse-only smoke test: drive the robot straight backward, then stop.

Run on the EV3. Verifies the wall-ball back-off motion still works in
isolation, independent of the vision/state-machine pipeline.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mov_control import drive_backward, stop

ROTATIONS = 1.0   # matches config.REVERSE_ROTATIONS (the wall-ball back-off)

print("Reversing {} rotations...".format(ROTATIONS))
drive_backward(ROTATIONS)
time.sleep(0.5)
stop()
print("Done — robot should have moved straight back ~{} rotations.".format(ROTATIONS))
