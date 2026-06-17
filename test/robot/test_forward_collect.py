#!/usr/bin/env python3
"""Drive forward, then collect a ball with claw and gate."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mov_control import drive_forward, drive_backward, stop
from claw_control import Claw
from gate_control import Gate
from ev3dev2.motor import MediumMotor, OUTPUT_B, OUTPUT_C


ROTATIONS = 1.0


# Setup
gate = Gate(MediumMotor, OUTPUT_C)
gate.setup()
print("Gate ready")

claw = Claw(MediumMotor, OUTPUT_B, gate)
print("Claw ready")
time.sleep(1)

# Drive
print("Driving forward {ROTATIONS} rotations...")
drive_forward(ROTATIONS)
time.sleep(0.5)

# Collect ball
print("Collecting ball...")
claw.collect_ball()
print("Done")
