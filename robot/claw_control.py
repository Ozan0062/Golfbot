import time

from ev3dev2.motor import SpeedDPS
from robot.gate_control import Gate
# LargeMotor maximum speed:  1050 deg/s
# MediumMotor maximum speed: 1560 deg/s

class Claw:
    def __init__(self, motor, port, Gate:Gate):
        self.motor = motor(port)
        self.gate = Gate

    def close_claw(self, speedDPS=-100):
        self.motor.on(speed=SpeedDPS(speedDPS))
        self.motor.wait_until('stalled')
        self.motor.stop(stop_action='hold')
        return True

    def open_claw(self, speedDPS=100):
        self.motor.on(speed=SpeedDPS(speedDPS))
        self.motor.wait_until('stalled')
        self.motor.stop(stop_action='hold')
        return True

    def reset_claw(self, speedDPS=100):
        self.motor.on(speed=SpeedDPS(speedDPS)) 
        self.motor.wait_until('stalled') 
        
        self.motor.stop(stop_action='hold') 
        self.motor.position = 0
        print("Claw is reset")
        return True

    def collect_ball(self):
        print("Start collect")
        print("Closing claw")
        self.close_claw()
        self.gate.gather_ball()
        print("Opening claw")
        self.open_claw()
        print("Collect done")
        return True
