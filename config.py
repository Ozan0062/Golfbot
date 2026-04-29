#all project constants

#Camera
CAMERA_INDEX = 1  #1usb,0pccam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640

#Field
FIELD_WIDTH_CM = 180.0
FIELD_HEIGHT_CM = 120.0

#yolo models
FIELD_MODEL_PATH = "models/best_field.onnx"
OBJECT_MODEL_PATH = "models/best_objects.onnx"
CONFIDENCE_THRESHOLD = 0.5

#object id
CLASS_NAMES = {
    0: "arrow",
    1: "cross",
    2: "ob",
    3: "robot",
    4: "wb",
}