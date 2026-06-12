import math
import cv2
import numpy as np
from vision.camera   import open_stream
from vision.field    import load_field_model, detect_corners, sort_corners, warp_field
from vision.detector import load_object_model, detect_objects, draw_detections
from vision.tracker  import pixels_to_cm, extract_objects, robot_px_to_cm
from vision.aruco    import create_detector, detect_robot, draw_robot
from controller.state_machine import GolfBotController
from vision.calibration import load_calibration, undistort
from config import ROBOT_FILTER_RADIUS_PX


def filter_near_robot(detections, robot_center_px, radius=ROBOT_FILTER_RADIUS_PX):
    """
    Remove ball detections whose pixel centre is within <radius> px of the
    robot's ArUco marker.  Avoids false positives caused by the marker itself.
    """
    if robot_center_px is None:
        return detections
    rx, ry = robot_center_px
    return [
        d for d in detections
        if d["class_name"] not in ("wb", "ob")
        or math.dist(d["center"], (rx, ry)) > radius
    ]


def main():
    print("GolfBot starting...")

    field_model    = load_field_model()
    object_model   = load_object_model()
    aruco_detector = create_detector()
    stream         = open_stream()
    mtx, dist      = load_calibration()
    controller     = GolfBotController()
    last_corners   = None

    while True:
        frame = stream.latest()
        if frame is None:
            continue   # thread not ready yet

        if mtx is not None:
            frame = undistort(frame, mtx, dist)

        # ── Field corners ────────────────────────────────────────────────────
        corners = detect_corners(field_model, frame)
        if len(corners) >= 4:
            last_corners = sort_corners(corners)

        if last_corners is None:
            print("Waiting for field corners...")
            cv2.imshow("GolfBot", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        warped, M = warp_field(frame, last_corners)
        h, w      = warped.shape[:2]

        # ── ArUco — detect on raw frame, project centre into warped coords ──
        robot_center_raw, robot_angle = detect_robot(aruco_detector, frame)
        if robot_center_raw is not None:
            pt = cv2.perspectiveTransform(
                np.array([[robot_center_raw]], dtype=np.float32), M
            )[0][0]
            robot_center = (float(pt[0]), float(pt[1]))
        else:
            robot_center = None

        # ── YOLO — detect objects, filter false hits near robot marker ───────
        detections = detect_objects(object_model, warped)
        detections = filter_near_robot(detections, robot_center)

        # ── Convert to cm and build world dict ───────────────────────────────
        world                = extract_objects(pixels_to_cm(detections, w, h))
        world["robot"]       = robot_px_to_cm(robot_center, w, h)
        world["robot_px"]    = robot_center
        world["robot_angle"] = robot_angle

        command = controller.update(world)

        # ── Debug overlay ────────────────────────────────────────────────────
        debug = draw_detections(warped, detections)
        debug = draw_robot(debug, robot_center, robot_angle)
        cv2.putText(debug, f"{controller.state.name}  {command.name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("GolfBot", debug)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    stream.stop()


if __name__ == "__main__":
    main()
