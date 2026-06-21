"""
detector_test.py — run object detection on a saved image.

Run:  python -m scripts.detector_test <image_path>
"""

import sys
import cv2

from vision.detector import load_object_model, detect_objects, draw_detections


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.detector_test <image_path>")
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
        print(f"  {det.class_name:12s} at ({det.center[0]:.0f}, {det.center[1]:.0f})  conf={det.confidence:.0%}")

    display = draw_detections(frame, detections)
    cv2.imshow("Detections", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
