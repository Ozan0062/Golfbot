"""
field_detect.py — live field-corner detection and warp preview.

Run:  python -m scripts.field_detect
Press 'w' to warp (saves warped_field.jpg), ESC to quit.
"""

import cv2

from vision.camera import open_camera, grab_frame, release
from vision.field import load_field_model, detect_corners, sort_corners, warp_field


def main():
    cap = open_camera()
    model = load_field_model()
    print("Field detection running. Press 'w' to warp, ESC to quit.")

    while True:
        frame = grab_frame(cap)
        display = frame.copy()

        corners = detect_corners(model, frame)

        # Draw detected corners
        for (cx, cy) in corners:
            cv2.circle(display, (int(cx), int(cy)), 8, (0, 255, 0), -1)

        if len(corners) >= 4:
            sorted_c = sort_corners(corners)
            # Draw the field outline
            pts = sorted_c.astype(int).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.putText(display, f"{len(corners)} corners found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Only {len(corners)} corners...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Field Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("w") and len(corners) >= 4:
            warped, _ = warp_field(frame, sort_corners(corners))
            cv2.imshow("Warped Field", warped)
            cv2.imwrite("warped_field.jpg", warped)
            print("Saved warped_field.jpg")

    release(cap)


if __name__ == "__main__":
    main()
