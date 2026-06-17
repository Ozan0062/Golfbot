"""
test_strategy_protocol.py - offline unit tests for PC-to-EV3 TCP command encoding.
Mocks socket transport; no EV3 brick or network connection required.
"""

import unittest
from unittest.mock import patch

from controller import ev3_controller


class EV3ProtocolStrategyTests(unittest.TestCase):
    """Offline protocol tests for the PC-to-EV3 command layer."""

    def test_drive_reverse_turn_collect_and_gate_commands_are_encoded_for_tcp(self):
        sent_commands = []

        def fake_send_recv(command):
            sent_commands.append(command)
            return "DONE"

        with patch.object(ev3_controller, "_send_recv", fake_send_recv):
            ev3_controller.drive(1.25)
            ev3_controller.reverse(0.5)
            ev3_controller.turn(2.0, "LEFT")
            ev3_controller.turn(3.0, "RIGHT")
            ev3_controller.collect()
            ev3_controller.release()
            ev3_controller.gate_open()
            ev3_controller.gate_close()

        self.assertEqual(
            sent_commands,
            [
                "FORWARD 1.25",
                "BACKWARD 0.5",
                "LEFT 2.0",
                "RIGHT 3.0",
                "COLLECT",
                "RELEASE",
                "GATE_OPEN",
                "GATE_CLOSE",
            ],
        )

    def test_stop_is_encoded_as_fire_and_forget_command(self):
        sent_commands = []

        with patch.object(ev3_controller, "_send", sent_commands.append):
            ev3_controller.stop()

        self.assertEqual(sent_commands, ["STOP"])

    def test_turn_rejects_invalid_direction_before_sending_to_robot(self):
        with patch.object(ev3_controller, "_send_recv") as send_recv:
            with self.assertRaises(ValueError):
                ev3_controller.turn(1.0, "SIDEWAYS")

        send_recv.assert_not_called()

    def test_send_recv_fails_safe_with_empty_reply_on_network_error(self):
        class FailingSocket:
            def __enter__(self):
                raise OSError("network unavailable")

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("controller.ev3_controller.socket.socket", return_value=FailingSocket()):
            reply = ev3_controller._send_recv("FORWARD 1.0")

        self.assertEqual(reply, "")

    def test_send_recv_uses_socket_success_path(self):
        class FakeSocket:
            def __init__(self):
                self.timeout = None
                self.connected_to = None
                self.sent = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def settimeout(self, timeout):
                self.timeout = timeout

            def connect(self, address):
                self.connected_to = address

            def sendall(self, payload):
                self.sent = payload

            def recv(self, size):
                return b"DONE"

        fake_socket = FakeSocket()

        with patch("controller.ev3_controller.socket.socket", return_value=fake_socket):
            reply = ev3_controller._send_recv("FORWARD 1.0")

        self.assertEqual(reply, "DONE")
        self.assertEqual(fake_socket.sent, b"FORWARD 1.0")
        self.assertEqual(fake_socket.connected_to, (ev3_controller.HOST, ev3_controller.PORT))

    def test_send_uses_socket_success_path(self):
        class FakeSocket:
            def __init__(self):
                self.connected_to = None
                self.sent = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def connect(self, address):
                self.connected_to = address

            def sendall(self, payload):
                self.sent = payload

        fake_socket = FakeSocket()

        with patch("controller.ev3_controller.socket.socket", return_value=fake_socket):
            ev3_controller._send("STOP")

        self.assertEqual(fake_socket.sent, b"STOP")
        self.assertEqual(fake_socket.connected_to, (ev3_controller.HOST, ev3_controller.PORT))


if __name__ == "__main__":
    unittest.main()
