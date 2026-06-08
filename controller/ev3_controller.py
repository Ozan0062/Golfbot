"""
ev3_controller.py — PC-side interface to the EV3 robot.

Holds a single persistent TCP connection to ev3_server.py on the brick.
Commands are newline-terminated strings sent over that connection.
If the link drops mid-run, _send/_send_recv reconnect automatically and retry.

COLLECT and RELEASE are blocking — they wait for the brick to reply "OK"
before returning, so the state machine knows the action finished.

Commands the brick must handle (ev3_server.py):
    FORWARD     — drive straight
    BACKWARD    — drive straight in reverse
    LEFT        — turn left in place
    RIGHT       — turn right in place
    STOP        — stop all drive motors
    GET_ANGLE   — reply with gyro angle (float)
    RESET_ANGLE — reset gyro to 0
    COLLECT     — run claw to pick up ball  (blocking on brick, replies OK)
    RELEASE     — open gate to release ball (blocking on brick, replies OK)
"""

import socket

HOST = "10.164.46.35"   # EV3 IP over WiFi
PORT = 5000

# Single socket kept alive for the whole run.
# None until the first command is sent.
_sock = None

RECV_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Connection management (internal)
# ---------------------------------------------------------------------------

def _connect():
    """Open a fresh TCP connection to the brick."""
    global _sock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(RECV_TIMEOUT_S)
    s.connect((HOST, PORT))
    _sock = s
    print(f"[EV3] Connected to {HOST}:{PORT}")


def _send(command: str):
    """Send a fire-and-forget command. Reconnects once if the link has dropped."""
    global _sock
    try:
        if _sock is None:
            _connect()
        _sock.sendall((command + "\n").encode())
    except OSError:
        print("[EV3] Connection lost — reconnecting...")
        _connect()
        _sock.sendall((command + "\n").encode())


def _send_recv(command: str) -> str:
    """Send a command and wait for the brick's reply. Returns '' on error."""
    global _sock
    try:
        if _sock is None:
            _connect()
        _sock.sendall((command + "\n").encode())
        return _sock.recv(1024).decode()
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
    """Turn counter-clockwise in place."""
    _send("LEFT")


def turn_right():
    """Turn clockwise in place."""
    _send("RIGHT")


def stop():
    """Stop all motors."""
    _send("STOP")


# ---------------------------------------------------------------------------
# Gyro
# ---------------------------------------------------------------------------

#def get_angle() -> float:
#    """Return accumulated gyro angle in degrees since last reset."""
#    return float(_send_recv("GET_ANGLE"))
#
#def reset_angle():
#    """Reset gyro to 0. Call once at startup before moving."""
#    _send("RESET_ANGLE")


# ---------------------------------------------------------------------------
# Claw & gate
# ---------------------------------------------------------------------------

def collect():
    """
    Close the claw to pick up a ball.
    Blocking — waits for the brick to reply OK before returning.
    """
    _send_recv("COLLECT")


def release():
    """
    Open the gate to release the ball at the goal.
    Blocking — waits for the brick to reply OK before returning.
    """
    _send_recv("RELEASE")
