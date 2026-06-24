"""
ev3_controller.py.

Sends string commands over TCP to ev3_server.py running on the brick.

Commands the brick must handle (ev3_server.py):
    FORWARD  <rot>   — drive forward <rot> motor rotations, reply "DONE"
    BACKWARD <rot>   — drive backward <rot> motor rotations, reply "DONE"
    LEFT     <rot>   — turn left <rot> motor rotations, reply "DONE"
    RIGHT    <rot>   — turn right <rot> motor rotations, reply "DONE"
    STOP             — stop all drive motors (fire-and-forget)
    COLLECT          — run claw to pick up ball (blocking on brick), reply "DONE"
    RELEASE          — open gate to release ball (blocking on brick), reply "DONE"
    GATE_OPEN        — open gate (blocking on brick), reply "DONE"
    GATE_CLOSE       — close gate (blocking on brick), reply "DONE"
"""

import socket

from golfbot_logger import get_logger

log = get_logger(__name__)

HOST = "10.210.93.35"   # EV3 IP over WiFi
PORT = 5000
RECV_TIMEOUT_S = 60.0   # Seconds to wait for a response before giving up (for blocking commands)


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
        log.error("EV3 command %r failed: %s", command, e)
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


def gate_open():
    """Open the gate. Blocking."""
    _send_recv("GATE_OPEN")


def gate_close():
    """Close the gate. Blocking."""
    _send_recv("GATE_CLOSE")
    

def reset_claw():
    """Reset the claw to its default position. Blocking."""
    _send_recv("RESET_CLAW")
    
def close_claw():
    """Close the claw. Blocking."""
    _send_recv("CLOSE_CLAW")
    
def gate_rotate():
    """Rotate the gate 360 degrees. Blocking."""
    _send_recv("GATE_ROTATE")