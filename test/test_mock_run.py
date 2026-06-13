"""
test_mock_run.py — Full state-machine test without the robot.

Run from project root:
    python -m test.test_mock_run

Patches ev3_controller so no TCP connection is needed.
Feeds the controller a scripted sequence of world dicts and prints every
decision alongside the [POSE] settle-window debug lines.

Scenario
--------
  Robot at (50, 60) cm, facing 90° (up in image coords).
  Ball at (100, 60) cm → bearing = atan2(0, 50) = 0°.
  angle_error(90, 0) = -90°  →  needs to turn LEFT.

Expected command sequence:
  1  TURN (LEFT, 90°)
  2-N  STOP  ← settle window, [POSE] Settling messages appear here
  N+1  FORWARD  ← fresh lock after settle, ball still far
  ...  STOP  ← settle window after drive
  M+1  COLLECT  ← ball now close (<25 px)
  ...  STOP  ← settle after collect, then route exhausted → REVERSE_WHITE
"""

import sys
import time
from unittest.mock import MagicMock

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Patch ev3_controller BEFORE importing anything from controller/
# ---------------------------------------------------------------------------

def _turn(rotations, direction):
    print(f"    [MOCK] turn {direction} {rotations:.2f} rot — sleeping 0.05s")
    time.sleep(0.05)

def _drive(rotations):
    print(f"    [MOCK] drive {rotations:.2f} rot — sleeping 0.05s")
    time.sleep(0.05)

def _reverse(rotations):
    print(f"    [MOCK] reverse {rotations:.2f} rot — sleeping 0.05s")
    time.sleep(0.05)

def _collect():
    print("    [MOCK] collect — sleeping 0.05s")
    time.sleep(0.05)

def _release():
    print("    [MOCK] release — sleeping 0.05s")
    time.sleep(0.05)

_mock_robot = MagicMock()
_mock_robot.turn.side_effect    = _turn
_mock_robot.drive.side_effect   = _drive
_mock_robot.reverse.side_effect = _reverse
_mock_robot.collect.side_effect = _collect
_mock_robot.release.side_effect = _release
sys.modules["controller.ev3_controller"] = _mock_robot

from controller.state_machine import GolfBotController, State
from controller.commands import Command


# ---------------------------------------------------------------------------
# World-dict helpers
# ---------------------------------------------------------------------------

ROBOT_POS  = (50.0,  60.0)   # cm
BALL_POS   = (100.0, 60.0)   # cm  — bearing from robot = 0°

ROBOT_PX_CENTER = (200, 240)
BALL_PX_FAR     = (400, 240)   # 200 px away  — triggers drive
BALL_PX_CLOSE   = (215, 242)   #  ~15 px away — triggers collect


def make_world(robot_angle, ball_px, robot_px=ROBOT_PX_CENTER, has_ball=True):
    return {
        "robot":           ROBOT_POS,
        "robot_px":        robot_px,
        "robot_angle":     robot_angle,
        "white_balls":     [BALL_POS]  if has_ball else [],
        "white_balls_px":  [ball_px]   if has_ball else [],
        "ob":              None,
        "ob_px":           None,
    }


# ---------------------------------------------------------------------------
# Scripted frames
# (sleep_s before feeding the frame, label, world-dict factory)
# ---------------------------------------------------------------------------

FRAMES = [
    # Robot sees ball, big angle error (90°) → TURN
    (0.00, "initial — 90° error",       lambda: make_world(90.0,  BALL_PX_FAR)),

    # Settle window ≈ 0.20s — pose cache should reject these
    (0.05, "settle  50ms",              lambda: make_world( 2.0,  BALL_PX_FAR)),
    (0.05, "settle 100ms",              lambda: make_world( 2.0,  BALL_PX_FAR)),
    (0.05, "settle 150ms",              lambda: make_world( 2.0,  BALL_PX_FAR)),
    (0.05, "settle 200ms",              lambda: make_world( 2.0,  BALL_PX_FAR)),

    # Fresh lock after settle — aligned (2° error < 10°), ball far → FORWARD
    (0.05, "post-turn, ball far",       lambda: make_world( 2.0,  BALL_PX_FAR)),
    (0.01, "same frame again",          lambda: make_world( 2.0,  BALL_PX_FAR)),

    # Settle after drive
    (0.05, "settle after drive  50ms",  lambda: make_world( 2.0,  BALL_PX_CLOSE)),
    (0.05, "settle after drive 100ms",  lambda: make_world( 2.0,  BALL_PX_CLOSE)),
    (0.05, "settle after drive 150ms",  lambda: make_world( 2.0,  BALL_PX_CLOSE)),
    (0.05, "settle after drive 200ms",  lambda: make_world( 2.0,  BALL_PX_CLOSE)),

    # Fresh lock — ball now close → COLLECT
    (0.05, "post-drive, ball close",    lambda: make_world( 2.0,  BALL_PX_CLOSE)),

    # Ball is now collected — remove it from world dicts
    (0.05, "settle after collect",      lambda: make_world( 2.0,  BALL_PX_CLOSE, has_ball=False)),
    (0.05, "settle after collect 2",    lambda: make_world( 2.0,  BALL_PX_CLOSE, has_ball=False)),
    (0.05, "settle after collect 3",    lambda: make_world( 2.0,  BALL_PX_CLOSE, has_ball=False)),
    (0.05, "settle after collect 4",    lambda: make_world( 2.0,  BALL_PX_CLOSE, has_ball=False)),
    (0.05, "no balls → REVERSE_WHITE",  lambda: make_world( 2.0,  BALL_PX_CLOSE, has_ball=False)),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run():
    print("\n" + "=" * 65)
    print("Mock state-machine test  (no robot required)")
    print("=" * 65 + "\n")

    ctrl = GolfBotController()

    for i, (sleep_s, label, factory) in enumerate(FRAMES):
        time.sleep(sleep_s)
        w   = factory()
        cmd = ctrl.update(w)
        print(f"  Frame {i+1:2d}  [{label:<32}]  "
              f"state={ctrl.state.name:<16}  cmd={cmd.name}")

        if ctrl.state in (State.DONE, State.REVERSE_WHITE):
            if ctrl.state == State.DONE:
                print("\n  ✓ Reached DONE.")
            break

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    run()
