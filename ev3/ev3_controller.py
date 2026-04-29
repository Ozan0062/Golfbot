"""
EV3 CONTROLLER

This module sends commands to the EV3 robot.
"""

import socket

HOST = "10.65.82.35"    # EV3 IP over USB (ev3dev)
PORT = 5000


def send_command(command):
    """Send a one-way command to the robot (no response expected)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())


def send_command_recv(command):
    """Send a command and return the robot's response as a string."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())
        response = s.recv(1024).decode()
    return response


# --- Gyro helpers ---

def get_angle():
    """Return the robot's current accumulated gyro angle in degrees."""
    return float(send_command_recv("GET_ANGLE"))


def get_speed():
    """Return the robot's current angular velocity in deg/s.
    NOTE: Do not call get_angle() and get_speed() in the same program —
    reading speed resets the gyro angle to zero.
    """
    return float(send_command_recv("GET_SPEED"))


def reset_angle(angle=0):
    """Reset the gyro angle to a given value (default 0)."""
    send_command(f"RESET_ANGLE:{angle}")


    # Future implementation:
    # socket communication
    # bluetooth communication