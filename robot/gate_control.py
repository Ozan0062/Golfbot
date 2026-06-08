import time
from ev3dev2.motor import SpeedDPS, SpeedPercent


class Gate:
    def __init__(self, motor, port):
        self.motor = motor(port)

    def setup(self):
        self.motor.command = 'reset'
        time.sleep(0.5)
        return True

    def _wait_stopped(self, timeout=2.0):
        """Block until the motor coast-stops (speed < 5 DPS), or force-stop on timeout."""
        start = time.time()
        while abs(self.motor.speed) > 5:
            if time.time() - start > timeout:
                self.motor.stop()
                break
            time.sleep(0.05)

    def _turn(self, degrees, speed=50):
        self.motor.stop()
        self._wait_stopped(timeout=2.0)     # wait for any previous motion to settle
        self.motor.on_for_degrees(
            speed=SpeedPercent(speed),
            degrees=degrees,
            brake=True,
            block=True
        )
        self._wait_stopped(timeout=5.0)     # wait for move to fully finish
        return True

    def rotate(self, rotations, speed=50):
        return self._turn(rotations * 360, speed)

    def open(self, speed=30):
        return self._turn(90, speed)

    def close(self, speed=30):
        return self._turn(-90, speed)
