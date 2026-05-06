"""
ev3_controller.py — PC-side interface to the EV3 robot.

Sends string commands over TCP to ev3_server.py running on the brick.
The robot team owns ev3_server.py and the command strings.
This file is the controller team's end of that contract.

Commands the brick must handle (ev3_server.py):
    FORWARD     — drive straight
    BACKWARD    — drive straight in reverse
    LEFT        — turn left in place
    RIGHT       — turn right in place
    STOP        — stop all drive motors
    GET_ANGLE   — reply with gyro angle (float)
    RESET_ANGLE — reset gyro to 0
    COLLECT     — run claw to pick up ball  (blocking on brick)
    RELEASE     — open gate to release ball (blocking on brick)
"""

import socket

HOST = "10.62.210.35"   # EV3 IP over WiFi
PORT = 5000


# ---------------------------------------------------------------------------
# Low-level transport (don't call these from state_machine.py)
# ---------------------------------------------------------------------------

def _send(command: str):
    """Send a fire-and-forget command to the brick."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())


RECV_TIMEOUT_S = 10.0   # max seconds to wait for brick to respond


def _send_recv(command: str) -> str:
    """Send a command and return the brick's response. Returns '' on timeout/error."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(RECV_TIMEOUT_S)
            s.connect((HOST, PORT))
            s.sendall(command.encode())
            return s.recv(1024).decode()
    except (socket.timeout, OSError) as e:
        print(f"[EV3] {command} timed out or failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def drive():
    """Drive straight forward."""
    _send("FORWARD")


def reverse():
    """Drive straight backward."""
    _send("BACKWARD")


def turn_left():
    """Turn left (counter-clockwise) in place."""
    _send("LEFT")


def turn_right():
    """Turn right (clockwise) in place."""
    _send("RIGHT")


def stop():
    """Stop all motors."""
    _send("STOP")


# ---------------------------------------------------------------------------
# Gyro
# ---------------------------------------------------------------------------

def get_angle() -> float:
    """Return accumulated gyro angle in degrees since last reset."""
    return float(_send_recv("GET_ANGLE"))


def reset_angle():
    """Reset gyro to 0. Call once at startup before moving."""
    _send("RESET_ANGLE")


# ---------------------------------------------------------------------------
# Claw & gate  (robot team must add handlers in ev3_server.py)
# ---------------------------------------------------------------------------

def collect():
    """
    Close the claw to pick up a ball.
    Blocking — waits for the brick to finish before returning.
    Requires: COLLECT handler in ev3_server.py using claw_control.py
    """
    _send_recv("COLLECT")   # recv forces us to wait for brick confirmation


def release():
    """
    Open the gate to release the ball at the goal.
    Blocking — waits for the brick to finish before returning.
    Requires: RELEASE handler in ev3_server.py using gate_control.py
    """
    _send_recv("RELEASE")   # recv forces us to wait for brick confirmation
