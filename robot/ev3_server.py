#!/usr/bin/env python3
import socket
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


def handle(command, conn):
    """Dispatch a single command string. Sends 'OK' back for blocking operations."""
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
    elif command == "COLLECT":
        claw.collect_ball()
        conn.sendall(b"OK\n")
    elif command == "RELEASE":
        gate.open()
        time.sleep(1)
        gate.close()
        conn.sendall(b"OK\n")
    else:
        print("Unknown command:", command)


# One persistent connection per PC session.
# Commands are newline-delimited so we can safely buffer across recv() calls.
# If the PC disconnects (e.g. WiFi drop), we stop the motors and wait for reconnect.
while True:
    conn, addr = server.accept()
    print("PC connected from", addr)
    buf = ""
    with conn:
        while True:
            try:
                data = conn.recv(1024)
            except OSError:
                break
            if not data:
                break   # clean disconnect
            buf += data.decode()
            # Process every complete (newline-terminated) command in the buffer.
            # Partial commands stay in buf until the next recv completes them.
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                command = line.strip()
                if command:
                    print("Received:", command)
                    handle(command, conn)
    # PC disconnected — stop motors so the robot doesn't keep driving
    stop()
    print("PC disconnected. Waiting for reconnect...")
