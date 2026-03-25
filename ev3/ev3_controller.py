"""
EV3 CONTROLLER

This module sends commands to the EV3 robot.
"""

import socket

HOST = "192.168.0.10"   # robot IP address
PORT = 5000


def send_command(command):

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode())


    # Future implementation:
    # socket communication
    # bluetooth communication