"""
camera_capture.py - capture training images from the overhead camera.

Run:  python -m scripts.camera_capture
Press SPACE to capture, ESC to quit. Saves to images-robot/, continuing the count.
"""

import os
import cv2

from vision.camera import open_camera, grab_frame, release
from config import CAMERA_WIDTH, CAMERA_HEIGHT

TOTAL_IMAGES = 100


def main():
    cap = open_camera()
    os.makedirs("images-robot", exist_ok=True)

    existing = [f for f in os.listdir("images-robot") if f.endswith(".jpg") and f[:-4].isdigit()]
    existing_count = max((int(f[:-4]) for f in existing), default=0)
    count  = existing_count
    target = existing_count + TOTAL_IMAGES
    if count:
        print(f"Resuming from image {count + 1}. {TOTAL_IMAGES} more to go.")
    print(f"Camera opened ({CAMERA_WIDTH}x{CAMERA_HEIGHT}). Press SPACE to capture, ESC to quit.")

    while count < target:
        frame = grab_frame(cap)
        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Aborted early.")
            break
        elif key == ord(" "):
            count += 1
            filename = os.path.join("images-robot", f"{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{count}/{target}] Saved {filename}")

    print(f"Done. {count} images saved to images-robot/")
    release(cap)


if __name__ == "__main__":
    main()
