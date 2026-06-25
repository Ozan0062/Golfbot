"""
tracker_pipeline.py - full vision pipeline smoke test on one frame.

Run:  python -m scripts.tracker_pipeline
Grabs one frame, runs field -> detect -> ArUco -> cm conversion, prints results.
"""

import sys
import cv2

from vision.camera import open_camera, grab_frame, release
from vision.field import load_field_model, detect_corners, sort_corners, warp_field
from vision.detector import load_object_model, detect_objects, draw_detections
from vision.aruco import create_detector, detect_robot, draw_robot
from vision.tracker import pixels_to_cm, extract_objects, robot_px_to_cm
from config import FIELD_WIDTH_CM, FIELD_HEIGHT_CM


def main():
    print("Loading models...")
    field_model = load_field_model()
    object_model = load_object_model()
    aruco_detector = create_detector()

    cap = open_camera()
    print("Grabbing frame...")
    frame = grab_frame(cap)
    release(cap)

    # Step 1: find field
    corners = detect_corners(field_model, frame)
    if len(corners) < 4:
        print(f"Only found {len(corners)} corners, need 4. Exiting.")
        sys.exit(1)

    sorted_c = sort_corners(corners)
    warped, M = warp_field(frame, sorted_c)
    h, w = warped.shape[:2]

    # Step 2: detect objects on warped image (YOLO)
    detections = detect_objects(object_model, warped)

    # Step 3: detect robot on warped image (ArUco)
    robot_center, robot_angle = detect_robot(aruco_detector, warped)

    # Step 4: convert to cm
    detections_cm = pixels_to_cm(detections, w, h)
    objects = extract_objects(detections_cm)

    robot_cm = robot_px_to_cm(robot_center, w, h)

    # Print results
    print(f"\n{'─' * 40}")
    print(f"Field warped to {w}x{h} px -> {FIELD_WIDTH_CM}x{FIELD_HEIGHT_CM} cm")
    print(f"{'─' * 40}")

    if robot_cm:
        print(f"  robot:        ({robot_cm[0]:.1f}, {robot_cm[1]:.1f}) cm  heading={robot_angle:.0f}°")
    else:
        print(f"  robot:        not found (no ArUco marker detected)")

    for key, val in objects.items():
        if key == "white_balls":
            print(f"  white_balls ({len(val)}):")
            for i, pos in enumerate(val):
                print(f"    [{i}] ({pos[0]:.1f}, {pos[1]:.1f}) cm")
        elif val:
            print(f"  {key:12s}: ({val[0]:.1f}, {val[1]:.1f}) cm")
        else:
            print(f"  {key:12s}: not found")

    # Show annotated image
    display = draw_detections(warped, detections)
    display = draw_robot(display, robot_center, robot_angle)
    cv2.imshow("Full Pipeline", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
