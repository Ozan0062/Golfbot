#all project constants

#Camera
CAMERA_INDEX = 1  #1usb,0pccam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640

#Field
FIELD_WIDTH_CM = 180.0
FIELD_HEIGHT_CM = 120.0

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

#ArUco marker
ARUCO_DICT = "DICT_4X4_50"
ARUCO_MARKER_ID = 0  # which marker ID is on the robot