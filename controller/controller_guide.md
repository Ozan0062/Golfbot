# GolfBot Controller — Guide

## What is the Controller?

The controller sits between two systems:

- **Input (Vision team):** `vision/tracker.py` → `extract_objects()` gives real-world positions in cm:
  ```python
  {
      "robot":       (x_cm, y_cm) or None,
      "white_balls": [(x_cm, y_cm), ...],
      "ob":          (x_cm, y_cm) or None,
      "cross":       (x_cm, y_cm) or None,
  }
  ```

- **Output (Robot team):** `controller/ev3_controller.py` → sends string commands over TCP:
  `FORWARD`, `LEFT`, `RIGHT`, `STOP`

Your job is to take positions in cm and decide which command to send, every frame.

---

## Design: Finite State Machine

The controller uses three states. The robot is always in exactly one of them.

```
IDLE ──► ALIGN ──► DRIVE
 ▲                   │
 └─── (ball collected)
```

| State | What the robot does | Transitions to |
|---|---|---|
| `IDLE` | Picks the nearest ball as target | `ALIGN` |
| `ALIGN` | Turns in place until facing the target | `DRIVE` when angle error < 10° |
| `DRIVE` | Drives forward toward the target | `IDLE` when within 15 cm, back to `ALIGN` if heading drifts |
| `DONE` | All balls collected, stops | — |

---

## Key Algorithms

### Angle to target
```python
import math

def angle_to_target(robot_pos, target_pos):
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    return math.degrees(math.atan2(dy, dx))
```

### Shortest turn direction
```python
def angle_error(current_angle, desired_angle):
    """Positive = turn RIGHT, Negative = turn LEFT."""
    return (desired_angle - current_angle + 180) % 360 - 180
```

### Nearest ball
```python
def nearest_ball(robot_pos, balls):
    if not balls:
        return None
    return min(balls, key=lambda b: math.dist(robot_pos, b))
```

---

## File Structure

```
Golfbot/
├── main.py                         ← entry point, wires everything together
├── config.py                       ← shared constants
├── robot/
│   ├── ev3_server.py               ← runs on the EV3 brick (robot team)
│   └── deploy.bat                  ← deploy robot code to brick
├── vision/                         ← vision team's files
│   ├── camera.py
│   ├── detector.py
│   ├── field.py
│   ├── tracker.py
│   ├── models/                     ← YOLO .onnx models
│   ├── training/                   ← training scripts
│   └── data/                       ← training images
├── controller/                     ← navigation logic (this folder)
│   ├── state_machine.py            ← FSM logic
│   ├── navigation.py               ← angle/distance math helpers
│   ├── ev3_controller.py           ← sends commands to robot over TCP
│   └── controller_guide.md        ← this file
└── test/
    └── test_connection.py          ← sanity check: camera + gyro
```

---

## Tuning Constants

At the top of `state_machine.py`:

```python
ALIGN_THRESHOLD_DEG  = 10   # how precisely to align before driving
ARRIVAL_THRESHOLD_CM = 15   # how close counts as "collected"
```

Adjust these on the real field. If the robot overshoots balls, lower `ARRIVAL_THRESHOLD_CM`. If it wastes time micro-adjusting heading, raise `ALIGN_THRESHOLD_DEG`.

---

## The Orientation Problem

The camera knows **where** the robot is but not **which way it's facing**. The gyro handles this — it tracks rotation from its reset point. At startup the gyro is reset to 0, so the direction the robot faces at launch becomes the 0° reference.

The gyro drifts slightly over many turns. For now this will not be calibrated during runtime.

---

## Interface Contract with Other Teams

**Vision team** delivers `extract_objects()` with at minimum:
- `"robot"` — robot position in cm, or `None`
- `"white_balls"` — list of white ball positions in cm
- `"ob"` — orange ball position in cm, or `None`

**Robot team** must handle: `FORWARD`, `LEFT`, `RIGHT`, `STOP` and the gyro commands `GET_ANGLE`, `RESET_ANGLE:<n>`.

---

## Testing Without the Full System

```python
# Run from project root: python -m test.test_connection
from controller.state_machine import GolfBotController
from unittest.mock import patch

with patch('controller.ev3_controller.send_command', side_effect=print), \
     patch('controller.ev3_controller.get_angle', return_value=0.0):

    ctrl = GolfBotController()
    world = {
        "robot":       (90.0, 60.0),
        "white_balls": [(30.0, 30.0), (150.0, 90.0)],
        "ob":          (10.0, 100.0),
    }
    for _ in range(30):
        ctrl.update(world)
```

---

## Common Pitfalls

**Coordinate system mismatch** — In the warped camera image, y increases downward. Confirm with the vision team which direction y goes, as it flips the sign of all angle calculations.

**Gyro drift** — Drifts over time, especially after many turns. Reset at startup while the robot is stationary and facing a known direction.

**Target lost mid-drive** — A ball may disappear from detections as the robot gets close. The controller holds the last known target position and keeps driving to it, so this is handled automatically.

**TCP flooding** — The camera loop runs fast. The robot only needs a command when something changes, not 30× per second. Consider only sending a command when it differs from the last one sent.
