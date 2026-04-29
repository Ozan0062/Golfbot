import socket
from pybricks.hubs import EV3Brick
from pybricks.parameters import Port, Direction
from pybricks.tools import wait
from pybricks.ev3devices import GyroSensor

HOST = ""
PORT = 5000

# --- Gyro sensor setup ---
# Change Port.S4 to whichever port the gyro is plugged into
gyro = GyroSensor(Port.S4, positive_direction=Direction.CLOCKWISE)
gyro.reset_angle(0)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Robot ready")

while True:

    conn, addr = server.accept()

    with conn:
        command = conn.recv(1024).decode()

        print("Received:", command)

        if command == "FORWARD":
            print("Move forward")

        elif command == "LEFT":
            print("Turn left")

        elif command == "RIGHT":
            print("Turn right")

        elif command == "STOP":
            print("Stop robot")

        # --- Gyro commands ---

        elif command == "GET_ANGLE":
            # Returns accumulated rotation angle in degrees
            angle = gyro.angle()
            conn.sendall(str(angle).encode())

        elif command == "GET_SPEED":
            # Returns angular velocity in deg/s
            # NOTE: Do not use GET_ANGLE and GET_SPEED in the same program —
            # reading speed resets the angle to zero.
            speed = gyro.speed()
            conn.sendall(str(speed).encode())

        elif command.startswith("RESET_ANGLE"):
            # Optional value after command, e.g. "RESET_ANGLE:0" or just "RESET_ANGLE"
            parts = command.split(":")
            target = int(parts[1]) if len(parts) > 1 else 0
            gyro.reset_angle(target)
            print(f"Gyro angle reset to {target}")