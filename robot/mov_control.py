from ev3dev2.motor import MoveTank, SpeedPercent, OUTPUT_A, OUTPUT_D

# ---------------------------------------------------------------------------
# Motor setup
# ---------------------------------------------------------------------------

tank = MoveTank(OUTPUT_D, OUTPUT_A)

DRIVE_SPEED_PCT = 40
TURN_SPEED_PCT  = 20

# Flip to -1 if motors are mounted in reverse
MOTOR_DIR = -1

# ---------------------------------------------------------------------------
# Public interface  (called from ev3_server.py)
# ---------------------------------------------------------------------------

def drive_forward(rotations: float):
    tank.on_for_rotations(SpeedPercent(MOTOR_DIR * DRIVE_SPEED_PCT), SpeedPercent(MOTOR_DIR * DRIVE_SPEED_PCT), rotations)

def drive_backward(rotations: float):
    tank.on_for_rotations(SpeedPercent(-MOTOR_DIR * DRIVE_SPEED_PCT), SpeedPercent(-MOTOR_DIR * DRIVE_SPEED_PCT), rotations)

def turn_left(rotations: float):
    # both wheels, opposite directions — left wheel back, right wheel forward
    tank.on_for_rotations(SpeedPercent(-TURN_SPEED_PCT), SpeedPercent(TURN_SPEED_PCT), rotations)

def turn_right(rotations: float):
    # both wheels, opposite directions — left wheel forward, right wheel back
    tank.on_for_rotations(SpeedPercent(TURN_SPEED_PCT), SpeedPercent(-TURN_SPEED_PCT), rotations)

def stop():
    tank.off(brake=True)
