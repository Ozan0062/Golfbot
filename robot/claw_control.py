from ev3dev2.motor import SpeedDPS

# LargeMotor maximum speed:  1050 deg/s
# MediumMotor maximum speed: 1560 deg/s

class Claw:
    def __init__(self, motor, port):
        self.motor = motor(port)

    def close_claw(self, speedDPS=20, position=30):
        self.motor.on_to_position(
            speed=SpeedDPS(speedDPS),
            position=position,
            brake=True,
            block=True
        )
        return True

    def open_claw(self, speedDPS=20, position=0):
        if self.motor.position >= 30:
            self.motor.on_to_position(
                speed=SpeedDPS(speedDPS),
                position=position,
                brake=True,
                block=True
            )
            return True
        return False

    def reset_claw(self, speedDPS=20, timeout_s=3):
        """Drive motor until stalled, then zero the position.
        timeout_s guards against infinite loop if stall is never detected."""
        import time
        self.motor.on(speed=SpeedDPS(speedDPS))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.motor.is_stalled:
                break
        self.motor.stop(stop_action='hold')
        self.motor.position = 0
        print("Claw is reset")
        return True

    def collect_ball(self):
        self.close_claw()
        self.open_claw()
        return True
