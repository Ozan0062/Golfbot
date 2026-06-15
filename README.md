# GolfBot

Autonomous golf ball collection robot. EV3 brick controlled from a PC via TCP, using an overhead camera with ArUco marker tracking and YOLO object detection.

**Field:** 180×120 cm. **Balls:** white (collect first) + orange (collect last). **Obstacle:** cross in the field centre.

## Architecture

```
Overhead Camera
      │
      ▼
 Vision Pipeline ──► world dict ──► State Machine ──► TCP ──► EV3 Brick
 (YOLO + ArUco)      (positions)     (decisions)              (motors)
```

The PC runs the vision pipeline and controller. The EV3 brick runs a socket server that executes motor commands.

## Project Structure

```
Golfbot/
├── main.py                          ← entry point, camera loop
├── config.py                        ← all shared constants
├── requirements.txt
│
├── vision/
│   ├── camera.py                    ← camera open/grab/release
│   ├── field.py                     ← field corner detection + perspective warp
│   ├── detector.py                  ← YOLO object detection (balls, cross)
│   ├── aruco.py                     ← ArUco marker detection (robot pose)
│   ├── tracker.py                   ← pixel→cm conversion, world dict assembly
│   ├── calibration.py               ← camera lens calibration
│   └── models/                      ← YOLO .onnx models
│
├── controller/
│   ├── state_machine.py             ← FSM: SEEK→AVOID→ALIGN→APPROACH→...→DONE
│   ├── navigation.py                ← angle math, path clearance, wall/corner geometry
│   ├── route_manager.py             ← ball ordering (Christofides TSP, white-first)
│   ├── ev3_controller.py            ← TCP commands to EV3
│   ├── calibration_manager.py       ← runtime drive/turn calibration
│   ├── calibration_tracker.py       ← EMA ratio tracking (px/rot, deg/rot)
│   ├── pose_cache.py                ← caches robot pose between detections
│   ├── tsp_christofides.py          ← 1.5-approx TSP solver
│   ├── commands.py                  ← Command enum
│   └── controller_guide.md          ← controller TLDR
│
├── robot/
│   ├── ev3_server.py                ← socket server on the EV3 brick
│   └── deploy.bat                   ← deploy to brick via SCP
│
└── test/                            ← hardware and integration tests
```

## Setup

```bash
git clone <repo-url> && cd Golfbot
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cd robot && deploy.bat (PS: maker)  # deploy code to EV3 brick
cd .. && python main.py
```

## How It Works

1. Camera grabs a frame
2. YOLO detects field corners → perspective warp to top-down 640×480 image
3. YOLO detects balls and cross, ArUco detects robot pose
4. Positions converted to cm, assembled into a `world` dict
5. State machine decides next action: turn, drive, collect, or avoid obstacle
6. Command sent to EV3 over TCP, motors execute

## State Machine

SEEK → lock target → check cross obstruction → check wall/corner → ALIGN → APPROACH → collect → SEEK

When no balls remain: REVERSE (scan) → DRIVE_GOAL → RELEASE → DONE

Cross obstacle avoidance and wall/corner approach both use staging waypoints — the robot drives to a safe position first, then approaches the ball from the correct angle.

## Configuration

All constants in `config.py`. Navigation thresholds at the top of `controller/state_machine.py` with inline comments.
