from controller.claw_control import Claw
from controller.gate_control import Gate
from ev3dev2.motor import MediumMotor, OUTPUTB, OUTPUTC

claw = Claw(MediumMotor, OUTPUTB)
gate = Gate(MediumMotor, OUTPUTC)

claw.reset_claw()
claw.collect_ball()

gate.open_gate()
gate.close_gate()