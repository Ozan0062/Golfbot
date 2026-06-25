"""
aruco.py - detect robot position and heading via its ArUco marker.

Standalone live test (also regenerates the printable marker):
    python -m scripts.aruco_test
"""

import sys
sys.path.append(".")

import cv2
import numpy as np

from config import ARUCO_DICT, ARUCO_MARKER_ID


# Map string name to OpenCV constant
_ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


def create_detector(dict_name=ARUCO_DICT):
    """Create an ArUco detector with the configured dictionary."""
    dict_id = _ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_robot(detector, frame, marker_id=ARUCO_MARKER_ID):
    """
    Detect the robot's ArUco marker in a frame.
    Returns (center, angle) or (None, None) if not found.
        center: (cx, cy) in pixels
        angle: heading in degrees (0 = right, 90 = down, etc.)
    """
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        return None, None

    # Find the specific marker
    for i, mid in enumerate(ids.flatten()):
        if mid == marker_id:
            pts = corners[i][0]  # 4 corner points of the marker
            center = pts.mean(axis=0)

            # Angle from top-left to top-right corner of marker
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            angle = np.degrees(np.arctan2(dy, dx))

            return (float(center[0]), float(center[1])), float(angle)

    return None, None


def draw_robot(frame, center, angle):
    """Draw robot position and heading arrow on frame (for debugging)."""
    if center is None:
        return frame

    display = frame.copy()
    cx, cy = int(center[0]), int(center[1])

    # Draw center point
    cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)

    # Draw heading arrow
    length = 50
    end_x = int(cx + length * np.cos(np.radians(angle)))
    end_y = int(cy + length * np.sin(np.radians(angle)))
    cv2.arrowedLine(display, (cx, cy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.1)

    # Label
    cv2.putText(display, f"robot {angle:.0f}deg", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return display


def generate_marker(marker_id=0, size=200, filename="aruco_marker.png"):
    """Generate and save a printable ArUco marker."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    # Add white border
    bordered = cv2.copyMakeBorder(marker, 40, 40, 40, 40,
                                   cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(filename, bordered)
    print(f"Saved marker ID {marker_id} to {filename} - print this and stick it on the robot")
    return bordered
