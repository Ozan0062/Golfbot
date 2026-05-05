# GolfBot

Autonomous golf ball collection robot using a Lego EV3 brick and overhead camera vision.

The project is split across three areas of responsibility:

- **robot/** — code that runs on the EV3 brick (motors + gyro)
- **vision/** — image recognition: field detection, ball/robot detection, coordinate mapping
- **controller/** — navigation logic: state machine, pathfinding, robot communication

---

## Project Structure

```
Golfbot/
├── main.py                         ← entry point
├── config.py                       ← shared constants (field size, model paths, camera index)
├── requirements.txt
│
├── robot/                          ← EV3 brick code (robot team)
│   ├── ev3_server.py               ← socket server, runs on the brick
│   └── deploy.bat                  ← deploys robot/ to the brick over SCP
|                                   ← insert new files here for movement etc.
│
├── vision/                         ← image recognition (vision team)
│   ├── camera.py                   ← camera open/grab/release
│   ├── field.py                    ← field corner detection + perspective warp
│   ├── detector.py                 ← YOLO object detection
│   ├── tracker.py                  ← pixel → cm conversion, object extraction
│   ├── models/                     ← YOLO .onnx model files
│   ├── training/                   ← model training scripts
│   └── data/                       ← training images
│
├── controller/                     ← navigation logic (controller team)
│   ├── state_machine.py            ← finite state machine (IDLE → ALIGN → DRIVE → DONE)
│   ├── navigation.py               ← angle/distance math helpers
│   ├── ev3_controller.py           ← sends commands to the brick over TCP
│   └── controller_guide.md        ← detailed guide for the controller
│
└── test/
    └── test_connection.py          ← sanity check: camera + robot connection
```

---

## Setup

```
1. Clone the repository
   git clone <repo-url>
   cd Golfbot

2. Create and activate a virtual environment
   python -m venv venv
   venv\Scripts\activate        (Windows)

3. Install dependencies
   pip install -r requirements.txt

4. Deploy code to the EV3 brick
   cd robot
   deploy.bat

5. Run
   python main.py
```

---

## How It Works

1. The overhead camera captures a frame.
2. `vision/field.py` detects the field corners and warps the image to a top-down view.
3. `vision/detector.py` runs YOLO to find the robot, white balls, orange ball, and cross marker.
4. `vision/tracker.py` converts pixel coordinates to real-world cm positions.
5. `controller/state_machine.py` receives these positions every frame and decides what command to send (`FORWARD`, `LEFT`, `RIGHT`, `STOP`).
6. `controller/ev3_controller.py` sends the command to the EV3 brick over TCP.
7. `robot/ev3_server.py` receives the command and drives the motors accordingly.

---

## Configuration

All constants are in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `1` | Camera device index (1 = USB, 0 = built-in) |
| `FIELD_WIDTH_CM` | `180.0` | Physical field width in cm |
| `FIELD_HEIGHT_CM` | `120.0` | Physical field height in cm |
| `CONFIDENCE_THRESHOLD` | `0.5` | YOLO detection confidence cutoff |

Navigation thresholds are in `controller/state_machine.py`:

| Constant | Default | Description |
|---|---|---|
| `ALIGN_THRESHOLD_DEG` | `10` | Max angle error before driving |
| `ARRIVAL_THRESHOLD_CM` | `15` | Distance at which a ball counts as collected |
