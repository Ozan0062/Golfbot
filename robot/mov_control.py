from ev3dev2.motor import MoveSteering, SpeedPercent, OUTPUT_A, OUTPUT_D

# ---------------------------------------------------------------------------
# Motor setup
# ---------------------------------------------------------------------------

steer = MoveSteering(OUTPUT_D, OUTPUT_A)

DRIVE_SPEED_PCT = 70
TURN_SPEED_PCT  = 50

# Flip to -1 if motors are mounted in reverse
MOTOR_DIR = -1

# ---------------------------------------------------------------------------
# Public interface  (called from ev3_server.py)
# ---------------------------------------------------------------------------

def drive_forward(rotations: float):
    steer.on_for_rotations(0, SpeedPercent(MOTOR_DIR * DRIVE_SPEED_PCT), rotations)
    steer.wait_until_not_moving(timeout=None)

def drive_backward(rotations: float):
    steer.on_for_rotations(0, SpeedPercent(-MOTOR_DIR * DRIVE_SPEED_PCT), rotations)
    steer.wait_until_not_moving(timeout=None)

def turn_left(rotations: float):
    steer.on_for_rotations(MOTOR_DIR * -100, SpeedPercent(MOTOR_DIR * TURN_SPEED_PCT), rotations)
    steer.wait_until_not_moving(timeout=None)

def turn_right(rotations: float):
    steer.on_for_rotations(MOTOR_DIR * 100, SpeedPercent(MOTOR_DIR * TURN_SPEED_PCT), rotations)
    steer.wait_until_not_moving(timeout=None)

def stop():
    steer.off(brake=True)
