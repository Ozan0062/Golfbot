from pybricks.ev3devices import UltrasonicSensor
from pybricks.parameters import Port

ultrasonic = UltrasonicSensor(Port.S1)

def read_distance():
    return ultrasonic.distance()