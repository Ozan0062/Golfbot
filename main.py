import math
import cv2
import numpy as np
from vision.camera   import open_stream
from vision.field    import load_field_model, detect_corners, sort_corners, warp_field
from vision.detector import load_object_model, detect_objects, draw_detections
from vision.tracker  import pixels_to_cm, extract_objects, robot_px_to_cm
from vision.aruco    import create_detector, detect_robot, draw_robot
from controller.state_machine import GolfBotController
from vision.calibration import load_calibration, build_undistort_maps, remap
from config import ROBOT_FILTER_RADIUS_PX, CAMERA_WIDTH, CAMERA_HEIGHT


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
    undist_maps    = None
    if mtx is not None:
        # Precompute remap tables once — much faster than cv2.undistort per frame
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        print(f"Lens calibration loaded — undistort maps built ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    controller     = GolfBotController()
    last_corners   = None

    while True:
        # Main loop tick
        frame = stream.latest()
        if frame is None:
            continue   # thread not ready yet

        if undist_maps is not None:
            frame = remap(frame, *undist_maps)

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

        # ── ArUco — detect on raw frame, project centre AND heading into warped coords
        robot_center_raw, robot_angle_raw = detect_robot(aruco_detector, frame)
        if robot_center_raw is not None:
            # Project a second point along the raw heading to transform the angle
            fwd_raw = (
                robot_center_raw[0] + 50 * math.cos(math.radians(robot_angle_raw)),
                robot_center_raw[1] + 50 * math.sin(math.radians(robot_angle_raw)),
            )
            pts = np.array([[robot_center_raw, fwd_raw]], dtype=np.float32)
            warped_pts = cv2.perspectiveTransform(pts, M)[0]

            robot_center = (float(warped_pts[0][0]), float(warped_pts[0][1]))
            robot_angle  = math.degrees(math.atan2(
                warped_pts[1][1] - warped_pts[0][1],
                warped_pts[1][0] - warped_pts[0][0],
            ))
        else:
            robot_center = None
            robot_angle  = None

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
