IMAGE RECOGNITION MODULE

This module handles all computer vision tasks.

Responsibilities:

* Read frames from the USB camera
* Process images
* Detect objects (ball, field, obstacles)
* Return coordinates of detected objects

Typical pipeline:

Camera Frame
↓
Image Processing
↓
Object Detection
↓
Return object coordinates

Future files in this module may include:

camera.py
detect_ball.py
detect_field.py
image_processing.py
