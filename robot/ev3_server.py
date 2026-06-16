#!/usr/bin/env python3
import socket
import threading
import time
from ev3dev2.motor import MediumMotor, OUTPUT_B, OUTPUT_C
from mov_control import drive_forward, drive_backward, turn_left, turn_right, stop
from claw_control import Claw
from gate_control import Gate

HOST = ""
PORT = 5000

gate = Gate(MediumMotor, OUTPUT_C)
gate.setup()

# ── gate calibration ──────────────────────────────────────────────────────────
print("=== Gate calibration ===")
print("  t <degrees>  - turn gate motor")
print("  set          - set this as zero and start server")
while True:
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not raw:
        continue
    parts = raw.split()
    if parts[0] == "t":
        try:
            gate._turn(float(parts[1]))
        except (IndexError, ValueError):
            print("  usage: t <degrees>")
    elif parts[0] == "set":
        gate.motor.command = "reset"
        time.sleep(0.3)
        print("  Gate zero set.")
        break
    else:
        print("  Unknown command.")
# ─────────────────────────────────────────────────────────────────────────────

claw = Claw(MediumMotor, OUTPUT_B, gate)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen()
print("Robot ready, listening on port", PORT)

claw.collect_ball()

while True:
    conn, addr = server.accept()
    with conn:
        command = conn.recv(1024).decode().strip()
        print("Received:", command)

        parts = command.split()
        cmd = parts[0]

        if cmd == "FORWARD":
            drive_forward(float(parts[1]))

        elif cmd == "BACKWARD":
            drive_backward(float(parts[1]))

        elif cmd == "LEFT":
            turn_left(float(parts[1]))

        elif cmd == "RIGHT":
            turn_right(float(parts[1]))

        elif cmd == "STOP":
            stop()

        elif cmd == "COLLECT":
            claw.collect_ball()
            conn.sendall(b"OK")

        elif cmd == "RELEASE":
            gate.open()
            time.sleep(1)
            gate.close()
            conn.sendall(b"OK")