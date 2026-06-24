"""config.py — all shared project constants."""

import os

# --- Camera ------------------------------------------------------------------
CAMERA_INDEX  = 1     # 1 = USB camera, 0 = built-in PC camera
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# --- Field -------------------------------------------------------------------
FIELD_WIDTH_CM  = 169.0
FIELD_HEIGHT_CM = 124.5

# Warped (top-down) image dimensions. NOTE: 900x600 is 3:2, which does NOT match
# the 170x124.5 field, so warped px/cm is anisotropic. Angle maths is done in cm
# (see controller/navigation.py) so this mismatch no longer biases headings.
WARPED_WIDTH  = 900
WARPED_HEIGHT = 600

# --- Parallax / height correction (cm) ---------------------------------------
CAMERA_HEIGHT_CM       = 178.0
ROBOT_MARKER_HEIGHT_CM = 19.8    # ArUco marker height above the field

# The point on the field the camera hangs directly above, in warped pixels.
# Scaled from (312, 303) in the old 640x480 view to 900x600.
# Re-measure in the 900x600 warped image if you need more precision.
CAMERA_CENTER_PX = (446, 350)

# --- Navigation safety -------------------------------------------------------
# Keep the robot centre at least this far from the field edges while collecting.
# Set to at least half the robot's widest dimension.
FIELD_SAFETY_MARGIN_CM = 15.0

# Goal: left wall, vertically centred.
GOAL_POSITION_CM = (0, FIELD_HEIGHT_CM / 2)
GOAL_POSITION_PX = (40, 300)   # claw target coordinate at the goal

# --- State-machine / navigation tuning ---------------------------------------
ALIGN_THRESHOLD_DEG = 2      # below this heading error we count as "aligned" and drive
MIN_TURN_ROTATIONS  = 0.25   # ignore turns smaller than this
TURN_DAMPING        = 0.6    # scale turns down to avoid oscillation when close

MARKER_TO_CLAW_CM = 16.8     # HORIZONTAL (floor-plane) offset from the ArUco marker
                             # centre to the claw tip, in cm. Measure floor-to-floor
                             # (point under the marker -> point under the claw tip),
                             # NOT the 3D slant from the 19.8 cm marker to the 7.5 cm tip.
CLAW_HEIGHT_CM    = 7.5      # claw tip height above the floor (cm). Informational only --
                             # the claw's floor position is derived geometrically from the
                             # marker, so no separate parallax step is applied to it.

COLLECT_RADIUS_CM = 5.0      # claw-tip -> ball distance (cm) at which we grab.
COLLECT_RADIUS_PX = 8        # legacy pixel radius (kept for tooling/tests; the live
                             # collect check is COLLECT_RADIUS_CM, measured in cm).
COLLECT_ANGLE_DEG = 5.0      # max angular offset (deg) in x AND y, measured from the
                             # marker using the arm length as reference, before grabbing.
COLLECT_NUDGE_MIN_PX = 3     # minimum drive distance (px) for an angle-correction nudge,
                             # so the motor command is always large enough to execute.

GOAL_ARRIVE_PX = 100         # arrive radius (px) for the final goal approach
GOAL_HEADING_DEG     = 180.0  # required robot heading when entering the goal
GOAL_HEADING_TOL_DEG = 5.0    # tolerance either side of GOAL_HEADING_DEG
REVERSE_ROTATIONS = 1.0      # how far to back up when no balls are visible
MAX_DRIVE_PX      = 80       # cap on drive distance per cycle, so we re-check often

CROSS_CLEARANCE_PX     = 70                       # stay at least this far from the cross centre
                                                  # cross is 10 cm radius ≈ 53 px + ~17 px robot buffer
AVOID_WAYPOINT_DIST_PX = CROSS_CLEARANCE_PX * 2  # how far to the side the dodge waypoint sits
AVOID_ARRIVE_PX        = 15                      # close enough to a waypoint to count as reached

# --- Cross pickup (ball sitting in/at the centre cross) ----------------------
# A ball this close to the cross is collected like a corner ball: staged
# approach along a fixed diagonal, then back off after grabbing.
CROSS_DIAMETER_CM       = 20.0   # physical size of the centre cross
# Fallback cross radius in warped px when the live detection size is unavailable.
# Use the larger per-axis px/cm scale so the radius is generous (never too small).
CROSS_RADIUS_PX         = CROSS_DIAMETER_CM / 2 * max(
    WARPED_WIDTH / FIELD_WIDTH_CM, WARPED_HEIGHT / FIELD_HEIGHT_CM)

WALL_MARGIN_PX      = 75    # a ball this close to a wall needs a staged approach
STAGING_DISTANCE_PX = 150    # standoff for the final straight-in approach.
                             # Must be >= WALL_MARGIN_PX / cos(45deg) ~= 170 so corner
                             # staging points land outside the margin on both axes.

# One staging point per wall/corner ball: 1x the staging distance.
CORNER_STAGE_DISTANCES_PX = (STAGING_DISTANCE_PX,)
FIELD_EDGE_MARGIN_PX = 30    # keep staging waypoints this far inside the field edges
GOAL_APPROACH_ANGLE_DEG = 180.0   # goal is on the left wall -> approach heading left

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
# These are starting values only — the live system refines them each run and,
# on ESC from main.py, writes the learned values back here (see
# controller/calibration_tracker.save_calibration_to_config). Turn calibration
# is tracked separately per direction because the robot can turn asymmetrically.
PIXELS_PER_ROTATION        = 63.32   # pixels travelled per motor rotation (measured)
DEGREES_PER_ROTATION_LEFT  = 28.33   # degrees turned per motor rotation, turning LEFT
DEGREES_PER_ROTATION_RIGHT = 34.47   # degrees turned per motor rotation, turning RIGHT

# Ignore ball detections within this radius of the robot (false positives).
ROBOT_FILTER_RADIUS_PX = 30   # pixels in the warped image

# --- ArUco marker ------------------------------------------------------------
ARUCO_DICT      = "DICT_4X4_50"
ARUCO_MARKER_ID = 0           # marker ID mounted on the robot
