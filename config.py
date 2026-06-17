"""config.py — all shared project constants."""

import os

# --- Camera ------------------------------------------------------------------
CAMERA_INDEX  = 1     # 1 = USB camera, 0 = built-in PC camera
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# --- Field -------------------------------------------------------------------
FIELD_WIDTH_CM  = 180.0
FIELD_HEIGHT_CM = 120.0

# Warped (top-down) image dimensions. 3:2 ratio matches the 180×120 field.
WARPED_WIDTH  = 900
WARPED_HEIGHT = 600

# --- Parallax / height correction (cm) ---------------------------------------
CAMERA_HEIGHT_CM       = 174.0
ROBOT_MARKER_HEIGHT_CM = 18.0    # ArUco marker height above the field

# The point on the field the camera hangs directly above, in warped pixels.
# Scaled from (312, 303) in the old 640×480 view to 900×600.
# Re-measure in the 900×600 warped image if you need more precision.
CAMERA_CENTER_PX = (451, 356)

# --- Navigation safety -------------------------------------------------------
# Keep the robot centre at least this far from the field edges while collecting.
# Set to at least half the robot's widest dimension.
FIELD_SAFETY_MARGIN_CM = 15.0

# Goal: left wall, vertically centred.
GOAL_POSITION_CM = (0, FIELD_HEIGHT_CM / 2)
GOAL_POSITION_PX = (0, WARPED_HEIGHT // 2)

# --- State-machine / navigation tuning ---------------------------------------
ALIGN_THRESHOLD_DEG = 2      # below this heading error we count as "aligned" and drive
MIN_TURN_ROTATIONS  = 0.25   # ignore turns smaller than this
TURN_DAMPING        = 0.6    # scale turns down to avoid oscillation when close

COLLECT_RADIUS_PX = 90       # close enough to a ball to grab it
GOAL_THRESHOLD_PX = 100      # close enough to the goal to stop and release
REVERSE_ROTATIONS = 1.5      # how far to back up when no balls are visible
MAX_DRIVE_PX      = 80       # cap on drive distance per cycle, so we re-check often

CROSS_CLEARANCE_PX     = 70                      # stay at least this far from the cross
AVOID_WAYPOINT_DIST_PX = CROSS_CLEARANCE_PX * 2  # how far to the side the dodge waypoint sits
AVOID_ARRIVE_PX        = 15                      # close enough to a waypoint to count as reached

WALL_MARGIN_PX      = 120    # a ball this close to a wall needs a staged approach
STAGING_DISTANCE_PX = 170    # standoff for the final straight-in approach.
                             # Must be >= WALL_MARGIN_PX / cos(45°) ≈ 170 so corner
                             # staging points land outside the margin on both axes.

# Two staging points per wall/corner ball: 2x then 1x the staging distance.
CORNER_STAGE_DISTANCES_PX = (STAGING_DISTANCE_PX * 2, STAGING_DISTANCE_PX)
FIELD_EDGE_MARGIN_PX = 30    # keep staging waypoints this far inside the field edges
GOAL_APPROACH_ANGLE_DEG = 180.0   # goal is on the left wall → approach heading left

# --- YOLO models -------------------------------------------------------------
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
FIELD_MODEL_PATH  = os.path.join(BASE_DIR, "vision", "models", "best_field.onnx")
OBJECT_MODEL_PATH = os.path.join(BASE_DIR, "vision", "models", "best_objects.onnx")
CONFIDENCE_THRESHOLD = 0.5

# Object class IDs (the robot is tracked via its ArUco marker, not YOLO).
CLASS_NAMES = {
    0: "cross",
    1: "ob",   # orange ball
    2: "wb",   # white ball
}

# --- Drive/turn calibration initial estimates (tune to your robot) -----------
PIXELS_PER_ROTATION  = 47.0   # pixels travelled per motor rotation (measured)
DEGREES_PER_ROTATION = 25.0   # degrees turned per motor rotation

# Ignore ball detections within this radius of the robot (false positives).
ROBOT_FILTER_RADIUS_PX = 30   # pixels in the warped image

# --- ArUco marker ------------------------------------------------------------
ARUCO_DICT      = "DICT_4X4_50"
ARUCO_MARKER_ID = 0           # marker ID mounted on the robot
