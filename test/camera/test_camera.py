import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
from vision.camera import open_camera, grab_frame, release

cap = open_camera()
print("Camera open. Press ESC to quit.")

while True:
    frame = grab_frame(cap)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

release(cap)
