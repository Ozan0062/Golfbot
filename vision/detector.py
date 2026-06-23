"""
detector.py — YOLO object detection (balls, cross) on the warped field image.

Standalone test on a saved image:
    python -m scripts.detector_test warped_field.jpg
"""

import math
import cv2
from ultralytics import YOLO
import sys
sys.path.append(".")
from config import OBJECT_MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_NAMES, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT
from controller.navigation import classify_zone

class Node_object:
    def __init__(self, class_name, center:tuple[float,float], size:tuple[float,float], dist_from_robot:float=0.0, confidence:float=0.0, class_id:int=-1, position_cm:tuple[float,float]=None):
        self.class_name = class_name
        self.center = center # coordinates in px
        self.size = size
        self.confidence = confidence
        self.class_id = class_id
        self.dist_from_robot = dist_from_robot

    def set_dist_from_robot(self, dist: float):
        self.dist_from_robot = dist

def load_object_model(path=OBJECT_MODEL_PATH):
    """Load object ONNX model"""
    return YOLO(path, task="detect")


def detect_objects(model, frame, conf=CONFIDENCE_THRESHOLD):
    """
    Run object detection on a (warped) field image.
    Returns list of dicts:
        {"class_id": int, "class_name": str, "center": (cx, cy), "size": (w, h), "confidence": float}
    """
    results = model.predict(frame, conf=conf, verbose=False)
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        xywh = box.xywh[0].cpu().numpy()
        det = Node_object(
            class_name=CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
            center=(float(xywh[0]), float(xywh[1])),
            size=(float(xywh[2]), float(xywh[3])),
            confidence=float(box.conf[0].item()),
            class_id=cls_id
        )
        detections.append(det)

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on a frame (for debugging)."""
    display = frame.copy()
    for det in detections:
        cx, cy = det.center
        w, h = det.size
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        color = {
            "cross": (0, 0, 255),
            "ob": (0, 165, 255),
            "wb": (255, 255, 255),
        }.get(det.class_name, (128, 128, 128))

        # We only draw boxes for the raw detections.
        # The coloured circles for zone classifications are drawn later by draw_world_objects
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        cv2.putText(display, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return display

def draw_world_objects(img, world):
    """Draw solid circles to visualize the WorldState lists."""
    for ball in world.white_balls_px:
        cv2.circle(img, (int(ball[0]), int(ball[1])), 8, (255, 255, 255), -1)
        _label(img, "open", (int(ball[0]) - 15, int(ball[1]) - 15), (255, 255, 255), scale=0.45)

    for ball in world.white_wall_balls_px:
        cv2.circle(img, (int(ball[0]), int(ball[1])), 8, (255, 0, 0), -1)
        _label(img, "wall", (int(ball[0]) - 15, int(ball[1]) - 15), (255, 0, 0), scale=0.45)

    for ball in world.white_corner_balls_px:
        cv2.circle(img, (int(ball[0]), int(ball[1])), 8, (0, 0, 255), -1)
        _label(img, "corner", (int(ball[0]) - 20, int(ball[1]) - 15), (0, 0, 255), scale=0.45)
        
    if world.ob_px:
        cv2.circle(img, (int(world.ob_px[0]), int(world.ob_px[1])), 8, (0, 165, 255), -1)
        
    if world.cross_px:
        cv2.circle(img, (int(world.cross_px[0]), int(world.cross_px[1])), 8, (0, 0, 255), -1)



# Colours are BGR (OpenCV order).
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_C_TARGET   = (255,   0, 255)   # magenta — the ball we're going for
_C_WAYPOINT = (245, 220,  30)   # cyan    — the waypoint we're driving to / will drive to
_C_ROBOT    = (  0, 255, 255)   # yellow  — the robot
_C_TEXT     = (245, 245, 245)   # near-white text on the banner

# What each state is doing, in plain language, plus the banner accent colour.
_STATE_INFO = {
    "SEEK":           ("Choosing the next ball",      (  0, 200,   0)),
    "AVOID":          ("Steering around an obstacle",  (  0, 165, 255)),
    "ALIGN":          ("Turning to face the ball",     (  0, 200,   0)),
    "APPROACH":       ("Driving to the ball",          (  0, 200,   0)),
    "REVERSE_WHITE":  ("Backing up to rescan",         (255, 180,   0)),
    "REVERSE_ORANGE": ("Backing up to rescan",         (255, 180,   0)),
    "DRIVE_GOAL":     ("Driving to the goal",          (255,   0, 255)),
    "RELEASE":        ("Releasing the balls",          (255,   0, 255)),
    "DONE":           ("Mission complete",             (180, 180, 180)),
}


def _dashed_line(img, p1, p2, color, thickness=2, dash=12, gap=9):
    """Draw a dashed line from p1 to p2 (OpenCV has no native dashed line)."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    steps = int(length // (dash + gap))
    for i in range(steps + 1):
        s = (i * (dash + gap)) / length
        e = min((i * (dash + gap) + dash) / length, 1.0)
        a = (int(x1 + (x2 - x1) * s), int(y1 + (y2 - y1) * s))
        b = (int(x1 + (x2 - x1) * e), int(y1 + (y2 - y1) * e))
        cv2.line(img, a, b, color, thickness)


def _label(img, text, org, color, scale=0.5, thickness=1):
    """Text with a thin black outline so it stays readable over any background."""
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_state_banner(img, state_name, command_name):
    """Top-left banner: big state name + plain-language action + the raw command."""
    action, accent = _STATE_INFO.get(state_name, ("", (0, 200, 0)))
    x, y, w, h = 10, 10, 360, 64
    panel = img[y:y + h, x:x + w]
    cv2.rectangle(panel, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.55, img[y:y + h, x:x + w], 0.45, 0, img[y:y + h, x:x + w])
    cv2.rectangle(img, (x, y), (x + 8, y + h), accent, -1)             # colour accent bar
    cv2.rectangle(img, (x, y), (x + w, y + h), accent, 1)              # thin border
    _label(img, state_name, (x + 20, y + 32), accent, scale=0.85, thickness=2)
    _label(img, action,     (x + 20, y + 54), _C_TEXT, scale=0.5, thickness=1)
    if command_name:
        _label(img, f"> {command_name}", (x + w - 130, y + 54), _C_TEXT, scale=0.5, thickness=1)


def _draw_staged_path(img, robot_center, avoid_target, next_waypoints, target_px):
    """
    Draw the waypoint the robot is driving to ("DRIVING TO") and the ones queued
    after it ("THEN"), chained robot → current → next... → target.
    """
    cur = (int(avoid_target[0]), int(avoid_target[1]))

    # Solid line from the robot to the waypoint it is driving to right now.
    if robot_center is not None:
        rc = (int(robot_center[0]), int(robot_center[1]))
        cv2.line(img, rc, cur, _C_WAYPOINT, 2, cv2.LINE_AA)

    # Dashed line chaining the upcoming waypoints, then on to the ball.
    chain = [cur] + [(int(w[0]), int(w[1])) for w in (next_waypoints or [])]
    if target_px is not None:
        chain.append((int(target_px[0]), int(target_px[1])))
    for a, b in zip(chain, chain[1:]):
        _dashed_line(img, a, b, _C_WAYPOINT, thickness=2)

    # The waypoint being driven to now — filled dot + ring + label.
    cv2.circle(img, cur, 7, _C_WAYPOINT, -1, cv2.LINE_AA)
    cv2.circle(img, cur, 15, _C_WAYPOINT, 2, cv2.LINE_AA)
    _label(img, "DRIVING TO", (cur[0] - 40, cur[1] - 20), _C_WAYPOINT, scale=0.5, thickness=1)

    # The waypoints that come after — hollow dots labelled THEN 1, THEN 2, ...
    for i, w in enumerate(next_waypoints or [], start=1):
        p = (int(w[0]), int(w[1]))
        cv2.circle(img, p, 11, _C_WAYPOINT, 2, cv2.LINE_AA)
        _label(img, f"THEN {i}", (p[0] - 28, p[1] - 16), _C_WAYPOINT, scale=0.45, thickness=1)


def _draw_legend(img):
    """Small key, bottom-left."""
    h = img.shape[0]
    items = [("robot", _C_ROBOT), ("driving to / next waypoint", _C_WAYPOINT), ("target ball", _C_TARGET)]
    y0 = h - 18 * len(items) - 10
    for i, (text, color) in enumerate(items):
        y = y0 + i * 18
        cv2.circle(img, (20, y), 6, color, -1, cv2.LINE_AA)
        _label(img, text, (34, y + 5), _C_TEXT, scale=0.45, thickness=1)


def draw_debug_overlay(warped, detections, world, robot_center, robot_angle,
                       state_name, command_name, locked_target=None,
                       avoid_target=None, next_waypoints=None):
    """
    Annotate the warped field for the live camera window.

    Shows, clearly: the current state + what the robot is doing (top banner),
    the robot, the locked target ball, and — when navigating — the waypoint
    being driven to now ("DRIVING TO") plus the ones queued after it ("THEN").

    avoid_target / next_waypoints are optional (pixel coords in the warped
    frame); pass them from GolfBotController.debug_view().
    """
    from vision.aruco import draw_robot

    debug = draw_detections(warped, detections)
    draw_world_objects(debug, world)
    debug = draw_robot(debug, robot_center, robot_angle)

    target_px = locked_target.px if locked_target is not None else None

    if target_px is not None:
        tx, ty = int(target_px[0]), int(target_px[1])
        cv2.rectangle(debug, (tx - 20, ty - 20), (tx + 20, ty + 20), _C_TARGET, 2)
        _label(debug, "TARGET", (tx - 28, ty + 38), _C_TARGET, scale=0.5, thickness=1)

        # Robot → target line + distance/heading readout (only useful with no waypoint detour).
        if robot_center is not None and robot_angle is not None and avoid_target is None:
            rx, ry = int(robot_center[0]), int(robot_center[1])
            cv2.rectangle(debug, (rx - 25, ry - 25), (rx + 25, ry + 25), _C_ROBOT, 2)
            cv2.line(debug, (rx, ry), (tx, ty), _C_ROBOT, 1, cv2.LINE_AA)
            dist_px = math.hypot(tx - rx, ty - ry)
            heading_error = (math.degrees(math.atan2(ty - ry, tx - rx)) - robot_angle + 180) % 360 - 180
            _label(debug, f"Dist {dist_px:.0f}px | Ang {heading_error:.1f}deg",
                   (rx - 50, ry - 35), _C_ROBOT, scale=0.45, thickness=1)

    if avoid_target is not None:
        _draw_staged_path(debug, robot_center, avoid_target, next_waypoints, target_px)

    _draw_state_banner(debug, state_name, command_name)
    _draw_legend(debug)
    return debug