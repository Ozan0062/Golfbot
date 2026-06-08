import time
from ev3dev2.motor import SpeedDPS, SpeedPercent


class Gate:
    def __init__(self, motor, port):
        self.motor = motor(port)

    def setup(self):
        self.motor.command = 'reset'
        time.sleep(0.5)
        return True

    def _turn(self, degrees, speed=50):
        self.motor.stop()
        start = time.time()
        while abs(self.motor.speed) > 5:
            if time.time() - start > 2.0:
                break
            time.sleep(0.05)

        self.motor.on_for_degrees(
            speed=SpeedPercent(speed),
            degrees=degrees,
            brake=True,
            block=True
        )

        start = time.time()
        while abs(self.motor.speed) > 5:
            if time.time() - start > 5.0:
                self.motor.stop()
                break
            time.sleep(0.05)

        return True

    def rotate(self, rotations, speed=50):
        return self._turn(rotations * 360, speed)

    def open(self, speed=30):
        return self._turn(90, speed)

    def close(self, speed=30):
        return self._turn(-90, speed)
