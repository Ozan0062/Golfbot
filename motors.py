from pybricks.ev3devices import Motor
from pybricks.parameters import Port

left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

def execute_action(action):

    if action == "SEARCH":
        left_motor.run(200)
        right_motor.run(-200)

    elif action == "AVOID":
        left_motor.run(-300)
        right_motor.run(-300)