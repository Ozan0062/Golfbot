# Movement: Switch from Time-Based to Rotation-Based

## Goal

Replace open-ended motor commands (caller uses `sleep()` to control duration) with commands that take a `rotations` parameter and block until complete.

Calibration (rotations → cm / degrees) is deferred — use a placeholder constant in `config.py` for now.

---

## Changes

### `robot/mov_control.py`

Replace `on()` calls with `on_for_rotations()`. Each function takes `rotations: float` and blocks until the motors finish.

```python
def drive_forward(rotations: float):
    tank.on_for_rotations(SpeedDPS(MOTOR_DIR * dps), SpeedDPS(MOTOR_DIR * dps), rotations)

def drive_backward(rotations: float):
    tank.on_for_rotations(SpeedDPS(-MOTOR_DIR * dps), SpeedDPS(-MOTOR_DIR * dps), rotations)

def turn_left(rotations: float):
    tank.on_for_rotations(SpeedDPS(-dps), SpeedDPS(dps), rotations)

def turn_right(rotations: float):
    tank.on_for_rotations(SpeedDPS(dps), SpeedDPS(-dps), rotations)
```

`stop()` stays unchanged.

---

### `robot/ev3_server.py`

Commands now carry rotations as a payload: `FORWARD:2.5`. Parse and pass through. Since `on_for_rotations` blocks, send `OK` after it returns so the controller knows the move is done.

```python
if command.startswith("FORWARD:"):
    rotations = float(command.split(":")[1])
    drive_forward(rotations)
    conn.sendall(b"OK")
```

Same pattern for `BACKWARD`, `LEFT`, `RIGHT`.

---

### `controller/ev3_controller.py`

Switch movement calls from fire-and-forget `_send` to blocking `_send_recv`, and accept a `rotations` argument.

```python
def drive(rotations: float):
    _send_recv(f"FORWARD:{rotations}")

def reverse(rotations: float):
    _send_recv(f"BACKWARD:{rotations}")

def turn_left(rotations: float):
    _send_recv(f"LEFT:{rotations}")

def turn_right(rotations: float):
    _send_recv(f"RIGHT:{rotations}")
```

---

### `test/test_movement.py`

Rewrite tests. Mock `ev3dev2` at import time (no hardware in CI). Tests to cover:

- `test_drive_forward_rotations` — verify `on_for_rotations` called with correct speeds and rotation count
- `test_drive_backward_rotations` — same for backward (check direction flip)
- `test_turn_left_rotations` — check opposite-direction speeds
- `test_turn_right_rotations` — same
- `test_zero_rotations` — `drive_forward(0)` should not raise
- `test_negative_rotations` — decide and document behaviour (raise `ValueError` or pass through)

---

## TCP / blocking design note

Each command opens its own short-lived TCP connection. The brick receives the command, runs `on_for_rotations` (blocking), sends `OK` when done, then closes the connection and loops back to `accept()`.

The connection only needs to survive the duration of the movement (a few seconds on local WiFi). This is the same pattern already used for `COLLECT` and `RELEASE` — no new risk. If a connection does drop, `_send_recv` returns `""` on the controller side; the state machine can treat that as "movement uncertain, re-check position."

---

## Callsites to update

Check `controller/navigation.py` and `controller/state_machine.py` for any calls to `drive()`, `reverse()`, `turn_left()`, `turn_right()` — all need a `rotations` argument added.

---

## Deferred: calibration

Add a placeholder to `config.py`:

```python
ROTATIONS_PER_CM  = 1.0   # TODO: measure and fill in
ROTATIONS_PER_DEG = 1.0   # TODO: measure and fill in
```

Navigation code uses these constants to convert distances/angles to rotations.
