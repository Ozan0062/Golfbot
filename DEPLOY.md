# GolfBot Deploy Guide

## Architecture

The PC is the brain. The EV3 brick is the muscle. Only one file lives on the robot.

```
PC                          EV3 Brick
──────────────────────      ──────────────────
main.py                     ev3_server.py
vision/                       │
controller/                   │
controller/ev3_controller.py ──TCP──►│ motors + gyro
```

---

## One-time setup

**1. Accept the SSH fingerprint** (first connection only)

```powershell
robot\deploy.bat
# type "yes" when prompted, then enter password: maker
```

---

## Every session

### Step 1 — Deploy (PowerShell, run once if you changed ev3_server.py)

```powershell
robot\deploy.bat
```

Password: `maker`

### Step 2 — Start the server (SSH terminal on the brick)

```bash
python3 ~/ev3_server.py
```

You should see:
```
Gyro angle: 0 deg
Robot ready, listening on port 5000
```

> Point the robot in your reference direction **before** running this.
> The gyro resets to 0 at startup — all angles are relative to this position.

### Step 3 — Run the program (PowerShell)

```powershell
python main.py
```

Press `ESC` in the OpenCV window to stop.

---

## Useful SSH commands

```bash
# Check what sensors are connected
cat /sys/class/lego-sensor/*/address

# Check what motors are connected
ls /sys/class/tacho-motor/

# Kill a running server if it froze
pkill -f ev3_server.py

# Restart the server
python3 ~/ev3_server.py
```

---

## Adding new robot commands

1. Add the command handler to `robot/ev3_server.py` on the PC
2. Run `robot\deploy.bat` to push it to the brick
3. Restart the server on the brick: `python3 ~/ev3_server.py`
4. Add the matching helper to `controller/ev3_controller.py` on the PC side

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection timed out` | Check IP in `ev3_controller.py` matches `hostname -I` on the brick |
| `DeviceNotFound: GyroSensor` | Gyro cable not seated — unplug and click back in, then restart server |
| `DeviceNotFound: LargeMotor` | Check motor ports B and C are plugged in |
| Gyro reading large number | Server wasn't restarted — kill and rerun `ev3_server.py` |
| Robot drives backwards | Flip the sign on both `run_forever` calls in the `FORWARD` block |
