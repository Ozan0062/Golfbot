"""Shared constants for the whole project."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Visual ---
CAMERA_INDEX  = 1     # 1 = USB, 0 = laptop
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

CAMERA_HEIGHT_CM = 176.5
CAMERA_CENTER_PX = (446, 350)

FIELD_WIDTH_CM  = 170.0
FIELD_HEIGHT_CM = 124.5

WARPED_WIDTH  = 900
WARPED_HEIGHT = 600

# --- Robot geometry ---
ROBOT_MARKER_HEIGHT_CM = 20.6
MARKER_TO_CLAW_CM      = 17     # floor distance, not the diagonal

# --- Driving / turning ---
ALIGN_THRESHOLD_DEG = 2
MIN_TURN_ROTATIONS  = 0.25
TURN_DAMPING        = 0.6
MAX_DRIVE_PX        = 80
REVERSE_ROTATIONS   = 1

COLLECT_RADIUS_CM = 2.0

# --- Wall / corner / cross approach ---
WALL_MARGIN_PX      = 100
STAGING_DISTANCE_PX = 150     # >= WALL_MARGIN_PX / cos(45deg) ~= 170
CORNER_STAGE_DISTANCES_PX = (STAGING_DISTANCE_PX,)
FIELD_EDGE_MARGIN_PX = 30

CROSS_DIAMETER_CM      = 20.0
CROSS_CLEARANCE_PX     = 100
AVOID_WAYPOINT_DIST_PX = CROSS_CLEARANCE_PX * 2
AVOID_ARRIVE_PX        = 15
CROSS_RADIUS_PX = CROSS_DIAMETER_CM / 2 * max(   # fallback when YOLO gives no box
    WARPED_WIDTH / FIELD_WIDTH_CM, WARPED_HEIGHT / FIELD_HEIGHT_CM)

# --- Goal: position and approach ---
GOAL_POSITION_PX = (860, 300)
GOAL_POSITION_CM = (GOAL_POSITION_PX[0] * FIELD_WIDTH_CM / WARPED_WIDTH,
                    GOAL_POSITION_PX[1] * FIELD_HEIGHT_CM / WARPED_HEIGHT)
GOAL_APPROACH_ANGLE_DEG = 0.0
GOAL_HEADING_DEG        = 0.0

# --- Goal: release ---
GOAL_RELEASE_MARKER_PX = (GOAL_POSITION_PX[0] - MARKER_TO_CLAW_CM * WARPED_WIDTH / FIELD_WIDTH_CM,
                          GOAL_POSITION_PX[1])
GOAL_RELEASE_X_TOL_PX        = 8
GOAL_RELEASE_LANE_TOL_PX     = 10
GOAL_RELEASE_HEADING_TOL_DEG = 2.0
GOAL_RELEASE_MAX_DRIVE_PX    = 40
GOAL_HEADING_MAX_CORRECTIONS            = 4
GOAL_HEADING_RECOVERY_REVERSE_ROTATIONS = 0.8
GOAL_LANE_MAX_REJECTIONS                = 2
GOAL_LANE_RECOVERY_REVERSE_ROTATIONS    = 0.8

# --- Zone calibration ---
# 4 quadrants, each self-tunes its px/rot and deg/rot; saved on clean exit (ESC).
ZONE_CENTER_PX        = (WARPED_WIDTH // 2, WARPED_HEIGHT // 2)
ZONE_CALIBRATION_FILE = os.path.join(BASE_DIR, "zone_calibration.json")
PIXELS_PER_ROTATION        = 63.21
DEGREES_PER_ROTATION_LEFT  = 30.81
DEGREES_PER_ROTATION_RIGHT = 29.27

# --- ArUco marker ---
ARUCO_DICT      = "DICT_4X4_50"
ARUCO_MARKER_ID = 0

# --- YOLO models ---
FIELD_MODEL_PATH     = os.path.join(BASE_DIR, "vision", "models", "best_field.onnx")
OBJECT_MODEL_PATH    = os.path.join(BASE_DIR, "vision", "models", "best_objects.onnx")
CONFIDENCE_THRESHOLD = 0.5
CLASS_NAMES = {0: "cross", 1: "ob", 2: "wb"}   # ob = orange, wb = white
ROBOT_FILTER_RADIUS_PX = 100
