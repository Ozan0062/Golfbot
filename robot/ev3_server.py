#!/usr/bin/env python3
import socket
import threading
import time
from ev3dev2.motor import MediumMotor, OUTPUT_A, OUTPUT_D
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_2
from mov_control import drive_forward, drive_backward, turn_left, turn_right, stop
from claw_control import Claw
from gate_control import Gate

HOST = ""
PORT = 5000

gyro = GyroSensor(INPUT_2)
gyro.mode   = 'GYRO-RATE'   # toggle to reset accumulated angle
gyro.mode   = 'GYRO-ANG'

# TODO robot team: set correct motor ports for claw and gate
claw = Claw(MediumMotor, OUTPUT_A)
gate = Gate(MediumMotor, OUTPUT_D)
gate.setup()

def print_gyro():
    while True:
        print("Gyro angle: " + str(gyro.value()) + " deg")
        time.sleep(5)

threading.Thread(target=print_gyro, daemon=True).start()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen()
print("Robot ready, listening on port", PORT)

while True:
    conn, addr = server.accept()
    with conn:
        command = conn.recv(1024).decode().strip()
        print("Received:", command)

        if command == "FORWARD":
            drive_forward()

        elif command == "BACKWARD":
            drive_backward()

        elif command == "LEFT":
            turn_left()

        elif command == "RIGHT":
            turn_right()

        elif command == "STOP":
            stop()

        elif command == "GET_ANGLE":
            conn.sendall(str(gyro.value()).encode())

        elif command == "GET_SPEED":
            gyro.mode = 'GYRO-RATE'
            conn.sendall(str(gyro.value()).encode())
            gyro.mode = 'GYRO-ANG'

        elif command == "RESET_ANGLE":
            gyro.mode = 'GYRO-RATE'
            gyro.mode = 'GYRO-ANG'
            print("Gyro reset")

        elif command == "COLLECT":
            claw.collect_ball()
            conn.sendall(b"OK")   # unblock the PC side

        elif command == "RELEASE":
            gate.open_gate()
            time.sleep(1)         # give ball time to roll out
            gate.close_gate()
            conn.sendall(b"OK")   # unblock the PC side
