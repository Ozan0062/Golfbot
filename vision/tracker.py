# vision/tracker.py — convert pixel detections to real-world cm coordinates
#
# Run for a full test (camera → field → detect → cm coords): python -m vision.tracker

import sys
sys.path.append(".")
from config import FIELD_WIDTH_CM, FIELD_HEIGHT_CM


def pixels_to_cm(detections, image_width, image_height,
                 field_w=FIELD_WIDTH_CM, field_h=FIELD_HEIGHT_CM):
    """
    Convert pixel coordinates from the warped field image to cm.
    Returns a new list of dicts with an added "position_cm" key: (x_cm, y_cm).
    """
    scale_x = field_w / image_width
    scale_y = field_h / image_height

    results = []
    for det in detections:
        cx, cy = det["center"]
        det_copy = dict(det)
        det_copy["position_cm"] = (cx * scale_x, cy * scale_y)
        results.append(det_copy)

    return results


def extract_objects(detections_cm):
    """
    Split detections into named objects for easy access.
    Returns dict: {
        "cross": (x, y) or None,
        "ob": (x, y) or None,
        "robot": (x, y) or None,
        "white_balls": [(x, y), ...],
    }
    """
    objects = {
    "cross": None,
    "ob": None,
    "robot": None,
    "arrow": None,
    "white_balls": [],
}

for det in detections_cm:
    name = det["class_name"]
    pos = det["position_cm"]

    if name == "wb":
        objects["white_balls"].append(pos)
    elif name == "arrow":
        # Use arrow as robot position (more precise center)
        objects["arrow"] = pos
    elif name in objects:
        if objects[name] is None or det["confidence"] > 0:
            objects[name] = pos

    return objects


# ── Standalone full pipeline test ───────────────────
# Grabs one frame, finds field, warps, detects objects, prints cm coords.
if __name__ == "__main__":
    from vision.camera import open_camera, grab_frame, release
    from vision.field import load_field_model, detect_corners, sort_corners, warp_field
    from vision.detector import load_object_model, detect_objects, draw_detections
    import cv2

    print("Loading models...")
    field_model = load_field_model()
    object_model = load_object_model()

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
    warped = warp_field(frame, sorted_c)
    h, w = warped.shape[:2]

    # Step 2: detect objects on warped image
    detections = detect_objects(object_model, warped)

    # Step 3: convert to cm
    detections_cm = pixels_to_cm(detections, w, h)
    objects = extract_objects(detections_cm)

    # Print results
    print(f"\n{'─' * 40}")
    print(f"Field warped to {w}x{h} px → {FIELD_WIDTH_CM}x{FIELD_HEIGHT_CM} cm")
    print(f"{'─' * 40}")
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
    cv2.imshow("Full Pipeline", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()