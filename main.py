import cv2
import numpy as np
from vision.camera   import open_camera, grab_frame, release
from vision.field    import load_field_model, detect_corners, sort_corners, warp_field
from vision.detector import load_object_model, detect_objects, draw_detections
from vision.tracker  import pixels_to_cm, extract_objects, robot_px_to_cm
from vision.aruco    import create_detector, detect_robot, draw_robot
from controller.state_machine import GolfBotController


def main():
    print("GolfBot starting...")

    field_model    = load_field_model()
    object_model   = load_object_model()
    aruco_detector = create_detector()
    cap            = open_camera()
    controller     = GolfBotController()

    while True:
        frame   = grab_frame(cap)
        corners = detect_corners(field_model, frame)
        if len(corners) < 4:
            continue

        warped, M = warp_field(frame, sort_corners(corners))
        h, w      = warped.shape[:2]

        # YOLO — balls, cross
        detections   = detect_objects(object_model, warped)
        world        = extract_objects(pixels_to_cm(detections, w, h))

        # ArUco — detect on raw frame (warping distorts the marker),
        # then project the center point into warped coordinates via M
        robot_center_raw, robot_angle = detect_robot(aruco_detector, frame)
        if robot_center_raw is not None:
            pt = cv2.perspectiveTransform(
                np.array([[robot_center_raw]], dtype=np.float32), M
            )[0][0]
            robot_center = (float(pt[0]), float(pt[1]))
        else:
            robot_center = None

        world["robot"]       = robot_px_to_cm(robot_center, w, h)
        world["robot_angle"] = robot_angle   # degrees, or None if not seen

        command = controller.update(world)

        # Debug overlay
        debug = draw_detections(warped, detections)
        debug = draw_robot(debug, robot_center, robot_angle)
        cv2.putText(debug, f"{controller.state.name}  {command}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("GolfBot", debug)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    release(cap)


if __name__ == "__main__":
    main()
