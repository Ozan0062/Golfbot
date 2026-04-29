# vision/detector.py — detect objects on the warped field image
#
# Run standalone to test on a saved image: python -m vision.detector warped_field.jpg

import cv2
from ultralytics import YOLO
import sys
sys.path.append(".")
from config import OBJECT_MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_NAMES


def load_object_model(path=OBJECT_MODEL_PATH):
    """Load objectONNX model"""
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
    "robot": (255, 0, 0),
    "arrow": (0, 255, 0),
    "wb": (255, 255, 255),
}.get(det["class_name"], (128, 128, 128))

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(display, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return display


# ── Standalone test ─────────────────────────────────
# Test on a saved image:  python -m vision.detector some_image.jpg
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