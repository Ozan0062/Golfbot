"""mov_control.py — drive/turn motor commands (runs on the EV3 brick)."""

from ev3dev2.motor import MoveSteering, SpeedPercent, OUTPUT_A, OUTPUT_D

# ---------------------------------------------------------------------------
# Motor setup
# ---------------------------------------------------------------------------

steer = MoveSteering(OUTPUT_D, OUTPUT_A)

DRIVE_SPEED_PCT = 70
TURN_SPEED_PCT  = 50

# Flip to -1 if motors are mounted in reverse.
MOTOR_DIR = -1


def _move(steering: int, speed_pct: float, rotations: float):
    """Run both motors at the given steering/speed for <rotations>, then block until stopped."""
    steer.on_for_rotations(steering, SpeedPercent(speed_pct), rotations)
    steer.wait_until_not_moving(timeout=None)


# ---------------------------------------------------------------------------
# Public interface  (called from ev3_server.py)
# ---------------------------------------------------------------------------

def drive_forward(rotations: float):
    _move(0, MOTOR_DIR * DRIVE_SPEED_PCT, rotations)

def drive_backward(rotations: float):
    _move(0, -MOTOR_DIR * DRIVE_SPEED_PCT, rotations)

def turn_left(rotations: float):
    _move(MOTOR_DIR * -100, MOTOR_DIR * TURN_SPEED_PCT, rotations)

def turn_right(rotations: float):
    _move(MOTOR_DIR * 100, MOTOR_DIR * TURN_SPEED_PCT, rotations)

def stop():
    steer.off(brake=True)
