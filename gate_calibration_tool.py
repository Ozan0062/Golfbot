#!/usr/bin/env python3
"""
gate_calibration_tool.py — interactive gate-motor calibration (runs on the EV3).

Commands:
    t <degrees>   turn the gate motor (e.g. t -30)
    set           mark the current position as zero and save
    q             quit
"""

import json, os, time

try:
    from ev3dev2.motor import MediumMotor, LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent
    ON_ROBOT = True
except ImportError:
    ON_ROBOT = False

PORT_MAP = {"A": OUTPUT_A, "B": OUTPUT_B, "C": OUTPUT_C, "D": OUTPUT_D}

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


class RealGate:
    def __init__(self, port="D"):
        port_const = PORT_MAP[port.upper()]
        for MotorClass in [MediumMotor, LargeMotor]:
            try:
                self.motor = MotorClass(port_const)
                print("  Motor found on OUTPUT_" + port.upper() + " (" + MotorClass.__name__ + ")")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("No motor found on OUTPUT_" + port.upper())
        self.motor.command = "reset"
        time.sleep(0.3)

    def turn(self, degrees):
        self.motor.on_for_degrees(SpeedPercent(40), degrees, brake=True, block=True)
        deadline = time.time() + 5
        while abs(self.motor.speed) > 5 and time.time() < deadline:
            time.sleep(0.05)

    def set_zero(self):
        self.motor.command = "reset"
        time.sleep(0.3)


class SimGate:
    def __init__(self):
        self._pos = 0

    def turn(self, degrees):
        self._pos += degrees
        print("  [SIM] motor at %+.1f deg" % self._pos)

    def set_zero(self):
        self._pos = 0


if ON_ROBOT:
    port = input("Gate motor port (A/B/C/D): ").strip() or "D"
    motor = RealGate(port)
else:
    motor = SimGate()
print("Gate calibration - " + ("ROBOT" if ON_ROBOT else "SIM"))
print("  t <degrees>  - turn motor")
print("  set          - mark current position as zero and save")
print("  q            - quit\n")

while True:
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not raw:
        continue

    parts = raw.split()

    if parts[0] == "q":
        break

    elif parts[0] == "t":
        try:
            deg = float(parts[1])
        except (IndexError, ValueError):
            print("  usage: t <degrees>")
            continue
        motor.turn(deg)

    elif parts[0] == "set":
        motor.set_zero()
        cal = {}
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE) as f:
                cal = json.load(f)
        cal["gate_zero_offset"] = True  # zero is now burned into the motor
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(cal, f, indent=2)
        print("  Zero set and saved.")

    else:
        print("  Unknown command.")
