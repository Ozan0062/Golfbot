import cv2
from vision.camera   import open_camera, grab_frame, release
from vision.field    import load_field_model, detect_corners, sort_corners, warp_field
from vision.detector import load_object_model, detect_objects
from vision.tracker  import pixels_to_cm, extract_objects
from controller.state_machine import GolfBotController


def main():
    print("GolfBot starting...")

    field_model  = load_field_model()
    object_model = load_object_model()
    cap          = open_camera()
    controller   = GolfBotController()

    while True:
        frame   = grab_frame(cap)
        corners = detect_corners(field_model, frame)
        if len(corners) < 4:
            continue

        warped     = warp_field(frame, sort_corners(corners))
        h, w       = warped.shape[:2]
        detections = detect_objects(object_model, warped)
        world      = extract_objects(pixels_to_cm(detections, w, h))

        command = controller.update(world)

        cv2.putText(warped, f"{controller.state.name}  {command}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("GolfBot", warped)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    release(cap)


if __name__ == "__main__":
    main()
