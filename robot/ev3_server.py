#!/usr/bin/env python3
import socket
import threading
import time
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_2

HOST = ""
PORT = 5000
DRIVE_SPEED = 30
TURN_SPEED  = 20

left_motor  = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)
gyro        = GyroSensor(INPUT_2)
gyro.mode   = 'GYRO-RATE'   # toggle to reset accumulated angle
gyro.mode   = 'GYRO-ANG'

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
            left_motor.run_forever(speed_sp=left_motor.max_speed * DRIVE_SPEED // 100)
            right_motor.run_forever(speed_sp=right_motor.max_speed * DRIVE_SPEED // 100)
        elif command == "LEFT":
            left_motor.run_forever(speed_sp=-(left_motor.max_speed * TURN_SPEED // 100))
            right_motor.run_forever(speed_sp=right_motor.max_speed * TURN_SPEED // 100)
        elif command == "RIGHT":
            left_motor.run_forever(speed_sp=left_motor.max_speed * TURN_SPEED // 100)
            right_motor.run_forever(speed_sp=-(right_motor.max_speed * TURN_SPEED // 100))
        elif command == "STOP":
            left_motor.stop()
            right_motor.stop()
        elif command == "GET_ANGLE":
            angle = gyro.angle()
            conn.sendall(str(angle).encode())
        elif command == "GET_SPEED":
            gyro.mode = 'GYRO-RATE'
            conn.sendall(str(gyro.value()).encode())
            gyro.mode = 'GYRO-ANG'
        elif command.startswith("RESET_ANGLE"):
            gyro.mode = 'GYRO-RATE'
            gyro.mode = 'GYRO-ANG'
            print("Gyro reset")
