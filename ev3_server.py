import socket

HOST = ""
PORT = 5000

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