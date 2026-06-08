#!/usr/bin/env python3
"""
test_claw.py - runs on the EV3 brick.
Tests close_claw, open_claw, and collect_ball.

Deploy and run:
    scp robot/claw_control.py robot@10.62.210.35:~/claw_control.py
    scp test/test_claw.py robot@10.62.210.35:~/test_claw.py
    ssh robot@10.62.210.35 python3 ~/test_claw.py
"""

import time
from ev3dev2.motor import MediumMotor, OUTPUT_B, OUTPUT_C
from robot.claw_control import Claw
from robot.gate_control import Gate

gate = Gate(MediumMotor, OUTPUT_C)
gate.setup()
claw = Claw(MediumMotor, OUTPUT_B, gate)

def wait(msg):
    print(msg)
    input("  -> press Enter to continue...")

print("=== Claw test ===")
print("Start position: " + str(claw.motor.position))

wait("Step 1: close_claw (should go DOWN to -30)")
result = claw.close_claw()
print("  returned: " + str(result))
print("  position: " + str(claw.motor.position))

time.sleep(0.5)

wait("Step 2: open_claw (should go back UP to 0)")
result = claw.open_claw()
print("  returned: " + str(result))
print("  position: " + str(claw.motor.position))

claw.reset_claw()

print("\n=== Done ===")