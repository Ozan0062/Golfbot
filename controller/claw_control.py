from ev3dev2.motor import SpeedDPS, SpeedRPS;

# Largemotor maximum speed is 1050 degrees per second
# Mediummotor maximum speed is 1560 degrees per second

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
       else:
           return False
      
    # Resets the claw and its position
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
