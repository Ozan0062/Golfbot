import time

from ev3dev2.motor import SpeedDPS
from gate_control import Gate
# LargeMotor maximum speed:  1050 deg/s
# MediumMotor maximum speed: 1560 deg/s

class Claw:
    def __init__(self, motor, port, gate:Gate):
        self.motor = motor(port)
        self.gate = gate

    def close_claw(self, speedDPS=-300, seconds=1):
        self.motor.on_for_seconds(
            speed=SpeedDPS(speedDPS),
            seconds=seconds,
            brake=True,
            block=True
        )
        self.motor.stop(stop_action='hold')
        return True

    def open_claw(self, speedDPS=300):
        self.motor.on(speed=SpeedDPS(speedDPS))
        self.motor.wait_until('stalled')
        self.motor.stop(stop_action='brake')
        return True

    def reset_claw(self, speedDPS=300):
        self.open_claw(speedDPS=speedDPS)
        time.sleep(0.5)
        self.motor.on(speed=SpeedDPS(speedDPS))
        self.motor.wait_until('stalled')
        self.motor.stop(stop_action='brake')
        self.motor.position = 0
        print("Claw is reset")
        return True

    def collect_ball(self):
        print("Start collect")
        print("Closing claw")
        self.close_claw()
        time.sleep(0.5)
        print("Rotating gate")
        self.gate.rotate(1)
        time.sleep(0.5)
        print("Closing claw again")
        self.close_claw()
        time.sleep(0.5)
        return True
