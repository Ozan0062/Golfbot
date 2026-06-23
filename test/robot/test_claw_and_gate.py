#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from claw_control import Claw
from gate_control import Gate
from ev3dev2.motor import MediumMotor, OUTPUT_B, OUTPUT_C
import time

gate = Gate(MediumMotor, OUTPUT_C)
gate.setup()
print("Gate ready")
time.sleep(1)

claw = Claw(MediumMotor, OUTPUT_B, gate)
print("Claw ready")
time.sleep(1)

print("Starting collect_ball...")
time.sleep(1)
claw.collect_ball()
sleep(0.5)
claw.reset_claw()
print("Done")