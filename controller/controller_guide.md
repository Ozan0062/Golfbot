# Controller Guide

See the flow diagram for detailed state machine logic.

## Overview

The controller receives a `world` dict from the vision pipeline every frame and sends motor commands to the EV3 over TCP. It's a finite state machine with 9 states.

## States

SEEK → AVOID → ALIGN → APPROACH → SEEK (loop per ball)
REVERSE_WHITE → REVERSE_ORANGE → DRIVE_GOAL → RELEASE → DONE (endgame)

## Key decisions in SEEK

1. Lock the nearest target ball (white balls first, then orange)
2. Cross blocking the path? → AVOID waypoint (perpendicular dodge)
3. Ball near a wall? → AVOID to staging point (perpendicular approach)
4. Ball in a corner? → AVOID to staging point (45° diagonal approach)
5. Path clear, open field → ALIGN directly

## Files

| File | Purpose |
|---|---|
| `state_machine.py` | FSM logic, all state transitions |
| `navigation.py` | Math helpers: angles, path clearance, zone classification, staging points |
| `route_manager.py` | Ball selection: nearest white first, then orange |
| `ev3_controller.py` | TCP commands to EV3 brick |
| `calibration_manager.py` | Runtime drive/turn calibration |
| `calibration_tracker.py` | EMA-based ratio tracking (px/rot, deg/rot) |
| `pose_cache.py` | Caches robot pose between ArUco detections |
| `commands.py` | Command enum |

## World dict

Built each frame by `main.py` from vision pipeline output:

```python
{
    "white_balls":    [(x_cm, y_cm), ...],
    "white_balls_px": [(x_px, y_px), ...],
    "ob":             (x_cm, y_cm) or None,
    "ob_px":          (x_px, y_px) or None,
    "cross":          (x_cm, y_cm) or None,
    "cross_px":       (x_px, y_px) or None,
    "robot_pos":      (x_cm, y_cm),
    "robot_px":       (x_px, y_px),
    "robot_angle":    float (degrees),
}
```

## Tuning

All thresholds are at the top of `state_machine.py` with comments explaining each one.
