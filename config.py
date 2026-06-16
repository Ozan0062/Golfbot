#all project constants

#Camera
CAMERA_INDEX = 1  #1usb,0pccam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640

#Field
FIELD_WIDTH_CM = 180.0
FIELD_HEIGHT_CM = 120.0

# Warped image dimensions (must match warp_field defaults)
WARPED_WIDTH  = 640
WARPED_HEIGHT = 480

# Navigation safety
# Robot centre must stay this far from field edges while collecting balls.
# Set this to at least half the robot's widest dimension.
FIELD_SAFETY_MARGIN_CM = 15.0

# Goal position — hardcoded: far right, vertically centered
GOAL_POSITION_CM = (FIELD_WIDTH_CM, FIELD_HEIGHT_CM / 2)
GOAL_POSITION_PX = (WARPED_WIDTH, WARPED_HEIGHT // 2)

#yolo models
FIELD_MODEL_PATH = "vision/models/best_field.onnx"
OBJECT_MODEL_PATH = "vision/models/best_objects.onnx"
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
