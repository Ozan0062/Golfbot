#all project constants

#Camera
CAMERA_INDEX = 1  #1usb,0pccam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

#Field
FIELD_WIDTH_CM = 180.0
FIELD_HEIGHT_CM = 120.0

# Warped image dimensions (3:2 ratio matches field 180×120)
WARPED_WIDTH  = 900
WARPED_HEIGHT = 600

# Height correction for parallax (cm)
CAMERA_HEIGHT_CM = 174.0
ROBOT_MARKER_HEIGHT_CM = 18.0       # ArUco marker height above field
# Kamera-center projiceret ned på banen (warped pixels).
# Det punkt på banen kameraet hænger direkte over (warped pixels).
# Skaleret fra (312, 303) i 640×480 → 900×600.
# MÅL GERNE IGEN i det nye 900×600 warped billede for præcision.
CAMERA_CENTER_PX = (439, 379)

# Navigation safety
# Robot centre must stay this far from field edges while collecting balls.
# Set this to at least half the robot's widest dimension.
FIELD_SAFETY_MARGIN_CM = 15.0

# Goal position — hardcoded: far right, vertically centered
GOAL_POSITION_CM = (FIELD_WIDTH_CM, FIELD_HEIGHT_CM / 2)
GOAL_POSITION_PX = (WARPED_WIDTH, WARPED_HEIGHT // 2)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#yolo models
FIELD_MODEL_PATH = os.path.join(BASE_DIR, "vision", "models", "best_field.onnx")
OBJECT_MODEL_PATH = os.path.join(BASE_DIR, "vision", "models", "best_objects.onnx")
CONFIDENCE_THRESHOLD = 0.5

#object id (no robot — detected via ArUco marker instead)
CLASS_NAMES = {
    0: "cross",
    1: "ob",  #orange ball
    2: "wb",  #white ball
}

# Drive/turn calibration initial estimates (tune these to your robot)
PIXELS_PER_ROTATION  = 47.0  # pixels the robot travels per motor rotation (measured)
DEGREES_PER_ROTATION = 25.0  # degrees the robot turns per motor rotation

# False-detection filter: ignore ball detections within this radius of the robot
ROBOT_FILTER_RADIUS_PX = 30  # pixels in warped (640×480) image

#ArUco marker
ARUCO_DICT = "DICT_4X4_50"
ARUCO_MARKER_ID = 0  # which marker ID is on the robot
