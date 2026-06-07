#!/usr/bin/env python3
from robot.claw_control import Claw
from robot.gate_control import Gate
from ev3dev2.motor import MediumMotor, OUTPUT_B, OUTPUT_C


gate = Gate(MediumMotor, OUTPUT_C)
gate.setup()

claw = Claw(MediumMotor, OUTPUT_B, gate)
claw.reset_claw()
claw.collect_ball()

