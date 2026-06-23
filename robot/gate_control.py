import time
from ev3dev2.motor import SpeedPercent

# 8 motor rotations = 360 degrees of gate
DEG_PER_ROTATION = 360.0 / 1.0


class Gate:
    def __init__(self, motor, port):
        self.motor = motor(port)

    def setup(self):
        self.motor.command = 'reset'
        time.sleep(0.5)
        return True

    def _run(self, rotations, speed=50):
        """Run motor for the given number of rotations."""
        self.motor.stop()
        start = time.time()
        while abs(self.motor.speed) > 5:
            if time.time() - start > 2.0:
                break
            time.sleep(0.05)

        degrees = rotations * 360.0
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

    def open(self, rotations=-0.5, speed=100):
        return self._run(rotations, speed)

    def close(self, rotations=-0.5, speed=100):
        return self._run(-rotations, speed)

    def rotate(self, rotations, speed=50):
        return self._run(-rotations, speed)