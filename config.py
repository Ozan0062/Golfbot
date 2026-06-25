"""All shared project constants."""
import os

# --- Camera ------------------------------------------------------------------
CAMERA_INDEX  = 1     # 1 = USB camera, 0 = built-in PC camera
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# --- Field -------------------------------------------------------------------
FIELD_WIDTH_CM  = 170.0
FIELD_HEIGHT_CM = 124.5

# Warped image size. 900x600 is 3:2, which doesn't perfectly match the field ratio.
# We do angle math in cm so this difference doesn't mess up headings.
WARPED_WIDTH  = 900
WARPED_HEIGHT = 600

# --- Parallax / height correction (cm) ---------------------------------------
CAMERA_HEIGHT_CM       = 176.5
ROBOT_MARKER_HEIGHT_CM = 20.6    # ArUco marker height above the field

# Camera center point in warped pixels.
# Re-measure this if we move the camera setup.
CAMERA_CENTER_PX = (446, 350)

# --- Navigation safety -------------------------------------------------------
# Keep the robot centre at least this far from the field edges while collecting.
# Set to at least half the robot's widest dimension.
FIELD_SAFETY_MARGIN_CM = 15.0

# Goal position. Treated like a ball on the right wall.
# We aim a bit inside the field so the robot doesn't crash into the wall during release.
GOAL_POSITION_PX = (860, 300)   # claw target coordinate at the goal
GOAL_POSITION_CM = (GOAL_POSITION_PX[0] * FIELD_WIDTH_CM / WARPED_WIDTH,
                    GOAL_POSITION_PX[1] * FIELD_HEIGHT_CM / WARPED_HEIGHT)
# Go to this point first before driving straight into the goal.
GOAL_LINEUP_PX = (680, GOAL_POSITION_PX[1])
GOAL_LINEUP_ARRIVE_PX = 8

# --- State-machine / navigation tuning ---------------------------------------
ALIGN_THRESHOLD_DEG = 2      # below this heading error we count as "aligned" and drive
MIN_TURN_ROTATIONS  = 0.25   # ignore turns smaller than this
TURN_DAMPING        = 0.6    # scale turns down to avoid oscillation when close

MARKER_TO_CLAW_CM = 17       # Distance from marker center to claw tip along the floor (cm).
                             # Don't measure the diagonal!
CLAW_HEIGHT_CM    = 7.5      # Height of claw tip from floor. Just for info.
GOAL_RELEASE_MARKER_PX = (
    GOAL_POSITION_PX[0] - MARKER_TO_CLAW_CM * WARPED_WIDTH / FIELD_WIDTH_CM,
    GOAL_POSITION_PX[1],
)
GOAL_RELEASE_X_TOL_PX = 8       # marker must be this close in x before opening gate
GOAL_RELEASE_LANE_TOL_PX = 10   # strict enough for the gate, loose enough for camera jitter
GOAL_RELEASE_HEADING_TOL_DEG = 2.0
GOAL_RELEASE_MAX_DRIVE_PX = 40  # short final steps so camera can re-check before release
GOAL_HEADING_MAX_CORRECTIONS = 4
GOAL_HEADING_RECOVERY_REVERSE_ROTATIONS = 0.8
GOAL_LANE_MAX_REJECTIONS = 2
GOAL_LANE_RECOVERY_REVERSE_ROTATIONS = 0.8

COLLECT_RADIUS_CM = 2.0      # Grab when claw tip is this close to the ball (cm).
COLLECT_RADIUS_PX = 8        # Old pixel radius for tests.
COLLECT_ANGLE_DEG = 5.0      # Max angle error before we try to grab.
COLLECT_NUDGE_MIN_PX = 3     # Drive at least this much when nudging so motors actually move.

GOAL_ARRIVE_PX = 100         # arrive radius (px) for the final goal approach
GOAL_HEADING_DEG     = 0.0    # required robot heading when entering the goal
GOAL_HEADING_TOL_DEG = 5.0    # tolerance either side of GOAL_HEADING_DEG
REVERSE_ROTATIONS = 1      # how far to back up when no balls are visible
MAX_DRIVE_PX      = 80       # cap on drive distance per cycle, so we re-check often

CROSS_CLEARANCE_PX     = 100                       # stay at least this far from the cross centre
                                                  # cross is 10 cm radius ≈ 53 px + ~17 px robot buffer
AVOID_WAYPOINT_DIST_PX = CROSS_CLEARANCE_PX * 2  # how far to the side the dodge waypoint sits
AVOID_ARRIVE_PX        = 15                      # close enough to a waypoint to count as reached

# --- Center cross pickup ----------------------
# Balls near the cross are picked up like corner balls (approach diagonally and back off).
CROSS_DIAMETER_CM       = 20.0   # Cross size in cm.
# Backup cross size in pixels if YOLO doesn't give us a bounding box.
CROSS_RADIUS_PX         = CROSS_DIAMETER_CM / 2 * max(
    WARPED_WIDTH / FIELD_WIDTH_CM, WARPED_HEIGHT / FIELD_HEIGHT_CM)

WALL_MARGIN_PX      = 100    # a ball this close to a wall needs a staged approach
STAGING_DISTANCE_PX = 150    # standoff for the final straight-in approach.
                             # Must be >= WALL_MARGIN_PX / cos(45deg) ~= 170 so corner
                             # staging points land outside the margin on both axes.

# Waypoints for wall/corner balls.
CORNER_STAGE_DISTANCES_PX = (STAGING_DISTANCE_PX,)
FIELD_EDGE_MARGIN_PX = 30    # keep staging waypoints this far inside the field edges
GOAL_APPROACH_ANGLE_DEG = 0.0     # goal is on the right wall -> approach heading right

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

# --- Default motor calibration ---
# These update automatically when you exit the program safely (ESC).
# Left and right turns have different values because the robot isn't perfectly symmetric.
PIXELS_PER_ROTATION        = 63.21   # pixels travelled per motor rotation (measured)
DEGREES_PER_ROTATION_LEFT  = 30.81   # degrees turned per motor rotation, turning LEFT
DEGREES_PER_ROTATION_RIGHT = 29.27   # degrees turned per motor rotation, turning RIGHT

# Ignore ball detections within this radius of the robot (false positives).
ROBOT_FILTER_RADIUS_PX = 100   # pixels in the warped image

# --- ArUco marker ------------------------------------------------------------
ARUCO_DICT      = "DICT_4X4_50"
ARUCO_MARKER_ID = 0           # marker ID mounted on the robot

# --- Zone calibration ---
# We split the field into 4 zones to handle uneven floor friction.
ZONE_CENTER_PX = (WARPED_WIDTH // 2, WARPED_HEIGHT // 2)   # (450, 300)
ZONE_CALIBRATION_FILE = os.path.join(BASE_DIR, "zone_calibration.json")
