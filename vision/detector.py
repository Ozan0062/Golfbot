# vision/detector.py — detect objects on the warped field image
#
# Run standalone to test on a saved image: python -m vision.detector warped_field.jpg

import math
import cv2
from ultralytics import YOLO
import sys
sys.path.append(".")
from config import OBJECT_MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_NAMES


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
        det = {
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
            "center": (float(xywh[0]), float(xywh[1])),
            "size": (float(xywh[2]), float(xywh[3])),
            "confidence": float(box.conf[0].item()),
        }
        detections.append(det)

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on a frame (for debugging)."""
    display = frame.copy()
    for det in detections:
        cx, cy = det["center"]
        w, h = det["size"]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        color = {
            "cross": (0, 0, 255),
            "ob": (0, 165, 255),
            "wb": (255, 255, 255),
        }.get(det["class_name"], (128, 128, 128))

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(display, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return display


def draw_debug_overlay(warped, detections, robot_center, robot_angle,
                       state_name, command_name, locked_target=None):
    """Draw detections, robot marker, and current state on the frame."""
    from vision.aruco import draw_robot

    debug = draw_detections(warped, detections)
    debug = draw_robot(debug, robot_center, robot_angle)

    if locked_target is not None:
        px = locked_target.px
        # Magenta square around target
        cv2.rectangle(debug, (int(px[0]) - 20, int(px[1]) - 20),
                      (int(px[0]) + 20, int(px[1]) + 20), (255, 0, 255), 2)

        if robot_center is not None and robot_angle is not None:
            # Yellow square around robot
            rx, ry = int(robot_center[0]), int(robot_center[1])
            cv2.rectangle(debug, (rx - 25, ry - 25), (rx + 25, ry + 25), (0, 255, 255), 2)

            # Calculate distance and angle error
            dx = px[0] - rx
            dy = px[1] - ry
            dist_px = math.hypot(dx, dy)
            target_heading = math.degrees(math.atan2(dy, dx))
            heading_error = (target_heading - robot_angle + 180) % 360 - 180

            # Draw line between them
            cv2.line(debug, (rx, ry), (int(px[0]), int(px[1])), (0, 255, 255), 1)

            # Show distance and angle near robot
            info_str = f"Dist: {dist_px:.0f}px | Ang: {heading_error:.1f}deg"
            cv2.putText(debug, info_str, (rx - 50, ry - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.putText(debug, f"{state_name}  {command_name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return debug


# ── Standalone test ─────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m vision.detector <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    model = load_object_model()
    detections = detect_objects(model, frame)

    print(f"\nFound {len(detections)} objects:")
    for det in detections:
        print(f"  {det['class_name']:12s} at ({det['center'][0]:.0f}, {det['center'][1]:.0f})  conf={det['confidence']:.0%}")

    display = draw_detections(frame, detections)
    cv2.imshow("Detections", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()