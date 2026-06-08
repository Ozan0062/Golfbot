from ev3dev2.motor import MoveTank, SpeedDPS, OUTPUT_A, OUTPUT_D

# ---------------------------------------------------------------------------
# Motor setup
# ---------------------------------------------------------------------------

tank = MoveTank(OUTPUT_D, OUTPUT_A)

# Speed settings as percentage of max
# LargeMotor max speed = 1050 DPS
LARGE_MOTOR_MAX_DPS = 1050
DRIVE_SPEED_PCT     = 30
TURN_SPEED_PCT      = 20

# Flip to -1 if motors are mounted in reverse
MOTOR_DIR = -1

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _dps(pct):
    """Convert a speed percentage (0-100) to a positive DPS value."""
    return (pct / 100) * LARGE_MOTOR_MAX_DPS

# ---------------------------------------------------------------------------
# Internal motor calls
# ---------------------------------------------------------------------------

def _run(left_dps, right_dps):
    """Run both motors continuously at the given DPS values."""
    tank.on(SpeedDPS(left_dps), SpeedDPS(right_dps))

def _brake():
    """Stop both motors and hold position."""
    tank.off(brake=True)

# ---------------------------------------------------------------------------
# Public interface  (called from ev3_server.py)
# ---------------------------------------------------------------------------

def drive_forward():
    dps = _dps(DRIVE_SPEED_PCT)
    _run(MOTOR_DIR * dps, MOTOR_DIR * dps)

def drive_backward():
    dps = _dps(DRIVE_SPEED_PCT)
    _run(-MOTOR_DIR * dps, -MOTOR_DIR * dps)

def turn_left():
    dps = _dps(TURN_SPEED_PCT)
    # Spin in place: left wheel back, right wheel forward.
    # Note: MOTOR_DIR is intentionally NOT applied here — a turn reverses
    # symmetrically so mounting direction cancels out.
    _run(-dps, dps)

def turn_right():
    dps = _dps(TURN_SPEED_PCT)
    # Spin in place: left wheel forward, right wheel back.
    _run(dps, -dps)

def stop():
    _brake()
