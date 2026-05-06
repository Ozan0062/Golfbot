from claw_control import Claw
from gate_control import Gate
from ev3dev2.motor import MediumMotor, OUTPUT_A, OUTPUT_D

claw = Claw(MediumMotor, OUTPUT_A)
gate = Gate(MediumMotor, OUTPUT_D)

claw.reset_claw()
claw.collect_ball()

gate.open_gate()
gate.close_gate()