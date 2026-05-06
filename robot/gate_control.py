from ev3dev2.motor import SpeedDPS

class Gate:
    def __init__(self, motor, port):
        self.motor = motor(port)

    def setup(self):
        self.motor.position = 0
        return True

    def open_gate(self, speedDPS=30, position=-90):
        self.motor.on_to_position(
            speed=SpeedDPS(speedDPS),
            position=position,
            brake=True,
            block=True
        )
        return True

    def close_gate(self, speedDPS=30, position=0):
        self.motor.on_to_position(
            speed=SpeedDPS(speedDPS),
            position=position,
            brake=True,
            block=True
        )
        return True
