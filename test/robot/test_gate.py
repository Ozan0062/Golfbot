#!/usr/bin/env python3
"""
test_gate.py - runs on the EV3 brick.
Tests gather_ball: should do exactly 1 rotation (360 degrees) and stop.

Deploy and run:
    scp robot/gate_control.py robot@<IP>:~/robot/gate_control.py
    scp test/test_gate.py robot@<IP>:~/test_gate.py
    ssh robot@<IP> python3 ~/test_gate.py
"""
import time
from ev3dev2.motor import MediumMotor, OUTPUT_C
from robot.gate_control import Gate


gate = Gate(MediumMotor, OUTPUT_C)

def divider(label):
    print("\n=== " + label + " ===")

def motor_state(label):
    print("[" + label + "]")
    print("  position: " + str(gate.motor.position))
    print("  speed:    " + str(gate.motor.speed))
    print("  state:    " + str(gate.motor.state))


divider("SETUP")
gate.setup()

divider("gather_ball #1 - should do 1 rotation (360 deg)")
motor_state("before")
gate.gather_ball(speed=50, rotations=1)
motor_state("after")
expected = 360
actual = gate.motor.position
diff = actual - expected
print("  expected position: " + str(expected))
print("  actual position:   " + str(actual))
print("  difference:        " + str(diff) + " degrees")

divider("waiting 3 seconds")
time.sleep(3)
motor_state("after sleep")

divider("gather_ball #2 - should do another 1 rotation")
motor_state("before")
gate.gather_ball(speed=50, rotations=5  )
motor_state("after")
print("  position since last reset: " + str(gate.motor.position))

divider("DONE")
