# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
test_turns.py - runs on the EV3 brick.

Test turning by manually entering direction and motor degrees.

Usage (on EV3 brick):
    python3 test_turns.py
"""

import sys
import time
import os

# Ensure we can import from the robot folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from robot.mov_control import turn_left, turn_right, stop
except ImportError:
    # Fallback if run directly inside robot folder
    try:
        from mov_control import turn_left, turn_right, stop
    except ImportError as e:
        print("Import error: %s" % e)
        print("Make sure mov_control.py is available.")
        sys.exit(1)

print("=" * 40)
print("  Turn Test (Motor Degrees)")
print("  Type direction (l/r) and degrees. Example: 'l 90' or 'r 180'")
print("  Type 'q' to quit")
print("=" * 40)

while True:
    try:
        raw = input("\nCommand (l/r deg) > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")
        break

    if not raw or raw == 'q' or raw == 'quit':
        break

    parts = raw.split()
    if len(parts) != 2:
        print("  Invalid format. Use: 'l 90' or 'r 180'")
        continue

    cmd = parts[0]
    
    try:
        degrees = float(parts[1])
    except ValueError:
        print("  Degrees must be a number.")
        continue

    # Convert motor degrees to motor rotations (what mov_control expects)
    rotations = degrees / 360.0

    if cmd in ('l', 'left'):
        print("  Turning LEFT %s degrees (%.2f rotations)..." % (degrees, rotations))
        turn_left(rotations)
    elif cmd in ('r', 'right'):
        print("  Turning RIGHT %s degrees (%.2f rotations)..." % (degrees, rotations))
        turn_right(rotations)
    else:
        print("  Unknown direction: '%s'. Use 'l' or 'r'." % cmd)

# Always stop motors when exiting
stop()
print("Exited cleanly.")
