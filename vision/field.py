# vision/field.py — detect field corners and warp to top-down view
#
# Run standalone to test field detection:
#   python -m vision.field

import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.append(".")
from config import FIELD_MODEL_PATH, CONFIDENCE_THRESHOLD


def load_field_model(path=FIELD_MODEL_PATH):
    """Load ONNX model for field corners."""
    return YOLO(path, task="detect")


def detect_corners(model, frame, conf=CONFIDENCE_THRESHOLD):
    """
    Run field model on a frame. Returns list of (cx, cy)
    """
    results = model.predict(frame, conf=conf, verbose=False)
    corners = []
    for box in results[0].boxes:
        # xywh gives [center_x, center_y, width, height]
        xywh = box.xywh[0].cpu().numpy()
        corners.append((float(xywh[0]), float(xywh[1])))
    return corners


def sort_corners(corners):
    """
    Sort 4 corner points into order: top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.array(corners, dtype=np.float32)

    #Handle only 4 corners
    while len(pts) > 4:
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        pts = np.delete(pts, np.argmin(dists), axis=0)

    if len(pts) != 4:
        raise ValueError(f"Expected 4 corners, got {len(pts)}")

    # Split into top/bottom by y, then left/right by x
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]

    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_field(frame, corners, output_width=640, output_height=480):
    """
    Perspective-warp the field region to a clean top-down rectangle.
    corners must be sorted: TL, TR, BR, BL.
    """
    dst = np.array([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (output_width, output_height))
    return warped


# ── Standalone test ─────────────────────────────────
#Opens camera, detects field corners live, draws them, warps on 'w' press.
if __name__ == "__main__":
    from vision.camera import open_camera, grab_frame, release

    cap = open_camera()
    model = load_field_model()
    print("Field detection running. Press 'w' to warp, ESC to quit.")

    while True:
        frame = grab_frame(cap)
        display = frame.copy()

        corners = detect_corners(model, frame)

        #Draw detected corners
        for (cx, cy) in corners:
            cv2.circle(display, (int(cx), int(cy)), 8, (0, 255, 0), -1)

        if len(corners) >= 4:
            sorted_c = sort_corners(corners)
            #Draw the field outline
            pts = sorted_c.astype(int).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.putText(display, f"{len(corners)} corners found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Only {len(corners)} corners...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Field Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("w") and len(corners) >= 4:
            warped = warp_field(frame, sort_corners(corners))
            cv2.imshow("Warped Field", warped)
            cv2.imwrite("warped_field.jpg", warped)
            print("Saved warped_field.jpg")

    release(cap)