"""
field.py - detect field corners and warp the field to a top-down view.

Standalone live test (detect corners, press 'w' to warp):
    python -m scripts.field_detect
"""

import sys
sys.path.append(".")

import cv2
import numpy as np
from ultralytics import YOLO

from config import FIELD_MODEL_PATH, CONFIDENCE_THRESHOLD, WARPED_WIDTH, WARPED_HEIGHT


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

    # Handle only 4 corners
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


def warp_field(frame, corners, output_width=WARPED_WIDTH, output_height=WARPED_HEIGHT):
    """
    Perspective-warp the field region to a clean top-down rectangle.
    corners must be sorted: TL, TR, BR, BL.
    Returns (warped_frame, M) where M is the homography matrix.
    """
    dst = np.array([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (output_width, output_height))
    return warped, M


def detect_field(field_model, frame, last_corners):
    """
    Detect field corners in the current frame.
    Returns updated corners (or the previous ones if detection fails this frame).
    """
    corners = detect_corners(field_model, frame)
    if len(corners) >= 4:
        return sort_corners(corners)
    return last_corners
