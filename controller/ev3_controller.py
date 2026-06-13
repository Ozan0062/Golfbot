"""
ev3_controller.py — PC-side interface to the EV3 robot.

Sends string commands over TCP to ev3_server.py running on the brick.

Commands the brick must handle (ev3_server.py):
    FORWARD  <rot>   — drive forward <rot> motor rotations, reply "DONE"
    BACKWARD <rot>   — drive backward <rot> motor rotations, reply "DONE"
    LEFT     <rot>   — turn left <rot> motor rotations, reply "DONE"
    RIGHT    <rot>   — turn right <rot> motor rotations, reply "DONE"
    STOP             — stop all drive motors (fire-and-forget)
    COLLECT          — run claw to pick up ball (blocking on brick), reply "DONE"
    RELEASE          — open gate to release ball (blocking on brick), reply "DONE"
"""

import socket

HOST = "10.233.49.35"   # EV3 IP over WiFi
PORT = 5000
RECV_TIMEOUT_S = 15.0   # Seconds to wait for a response before giving up (for blocking commands)


# ---------------------------------------------------------------------------
# Low-level transport
# ---------------------------------------------------------------------------

def _send(command: str):
    """
    Fire-and-forget used by STOP command, which should interrupt any ongoing motion immediately.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())


def _send_recv(command: str) -> str:
    """ 
    Send command and block until brick replies. Returns '' on error.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(RECV_TIMEOUT_S)
            s.connect((HOST, PORT))
            s.sendall(command.encode())
            return s.recv(1024).decode()
    except (socket.timeout, OSError) as e:
        print(f"[EV3] {command!r} failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Movement  (all blocking — wait for brick to finish)
# ---------------------------------------------------------------------------

def drive(rotations: float):
    """Drive straight forward by <rotations> motor rotations. Blocking."""
    _send_recv(f"FORWARD {rotations}")


def reverse(rotations: float):
    """Drive straight backward by <rotations> motor rotations. Blocking."""
    _send_recv(f"BACKWARD {rotations}")


def turn(rotations: float, direction: str = "LEFT"):
    """
    Turn in place by <rotations> motor rotations. Blocking.
    direction: "LEFT" (counter-clockwise) or "RIGHT" (clockwise)
    """
    if direction not in ("LEFT", "RIGHT"):
        raise ValueError(f"direction must be 'LEFT' or 'RIGHT', got {direction!r}")
    _send_recv(f"{direction} {rotations}")


def stop():
    """Stop all motors immediately (fire-and-forget)."""
    _send("STOP")


# ---------------------------------------------------------------------------
# Claw & gate
# ---------------------------------------------------------------------------

def collect():
    """Close claw to pick up a ball. Blocking."""
    _send_recv("COLLECT")


def release():
    """Open gate to release ball at goal. Blocking."""
    _send_recv("RELEASE")
