from ev3dev2.motor import SpeedDPS

# LargeMotor maximum speed:  1050 deg/s
# MediumMotor maximum speed: 1560 deg/s

class Claw:
    def __init__(self, motor, port):
        self.motor = motor(port)

    def close_claw(self, speedDPS=40, position=-50):
        self.motor.on_to_position(
            speed=SpeedDPS(speedDPS),
            position=position,
            brake=True,
            block=True
        )
        return True

    def open_claw(self, speedDPS=40, position=0):
        self.motor.on_to_position(
                speed=SpeedDPS(speedDPS),
                position=position,
                brake=True,
                block=True
            )
        return True

    def reset_claw(self, speedDPS=20):
        self.motor.on(speed=SpeedDPS(speedDPS)) 
        while self.motor.speed > 1:
            pass 
        
        self.motor.stop(stop_action='hold') 
        self.motor.position = 0
        print("Claw is reset")
        return True

    def collect_ball(self):
        self.close_claw()
        self.open_claw()
        return True
