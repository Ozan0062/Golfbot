"""
ev3_controller.py — PC-side interface to the EV3 robot.

Sends string commands over TCP to ev3_server.py running on the brick.
The robot team owns ev3_server.py and the command strings.
This file is the controller team's end of that contract.
"""

import socket

HOST = "10.65.82.35"   # EV3 IP over USB (ev3dev)
PORT = 5000


# ---------------------------------------------------------------------------
# Low-level transport (don't call these from state_machine.py)
# ---------------------------------------------------------------------------

def _send(command: str):
    """Send a fire-and-forget command to the brick."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())


def _send_recv(command: str) -> str:
    """Send a command and return the brick's response."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())
        return s.recv(1024).decode()


# ---------------------------------------------------------------------------
# Movement commands  (robot team fills in ev3_server.py handler for each)
# ---------------------------------------------------------------------------

def drive():
    """Drive straight forward."""
    _send("FORWARD")


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
# Gyro  (robot team owns the sensor; we just read it)
# ---------------------------------------------------------------------------

def get_angle() -> float:
    """Return accumulated gyro angle in degrees since last reset."""
    return float(_send_recv("GET_ANGLE"))


def reset_angle():
    """Reset gyro to 0. Call once at startup before moving."""
    _send("RESET_ANGLE")
