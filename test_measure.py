"""
test_measure.py — live camera tool for checking robot pose and ball measurements.

Displays on the warped top-down view:
  - Robot marker dot + heading arrow
  - Cyan dot at the height-corrected claw tip position
  - For every known ball: claw-tip → ball distance in px and degrees off heading

Claw tip pipeline:
  1. Detect ArUco marker in raw frame → project to warped coords (uncorrected)
  2. Project forward from raw marker position by MARKER_TO_CLAW_PX along heading
  3. Apply correct_robot_height at CLAW_HEIGHT_CM (7.5) to get true floor position

This gives a heading-independent claw→ball distance.

Run:
    python test_measure.py

ESC to quit.
"""

import math
import cv2

from golfbot_logger import setup_logging, get_logger
from vision.camera           import open_stream
from vision.field            import load_field_model, warp_field, detect_field
from vision.detector         import load_object_model, detect_objects
from vision.tracker          import (detect_robot_pose_in_warped_coords,
                                     correct_robot_height,
                                     filter_detections_near_robot)
from vision.aruco            import create_detector
from vision.lens_calibration import load_calibration, build_undistort_maps, undistort_frame
from controller.navigation   import angle_to_target, angle_error
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_CENTER_PX, CAMERA_HEIGHT_CM,
    ROBOT_MARKER_HEIGHT_CM,
    FIELD_WIDTH_CM, FIELD_HEIGHT_CM,
    WARPED_WIDTH,
    MARKER_TO_CLAW_CM, CLAW_HEIGHT_CM,
)

# ── Physical measurements ─────────────────────────────────────────────────────
BALL_HEIGHT_CM = 0.0    # golf ball parallax correction (negligible)

PX_PER_CM         = WARPED_WIDTH / FIELD_WIDTH_CM          # 5.0 px/cm
MARKER_TO_CLAW_PX = MARKER_TO_CLAW_CM * PX_PER_CM

MERGE_RADIUS_PX   = 40    # max px shift for same-ball detection across frames

setup_logging()
log = get_logger(__name__)

log.info("MARKER_TO_CLAW_CM=%.1f  CLAW_HEIGHT_CM=%.1f  (%.1f px forward)",
         MARKER_TO_CLAW_CM, CLAW_HEIGHT_CM, MARKER_TO_CLAW_PX)

# ── colours ───────────────────────────────────────────────────────────────────
C_ROBOT        = (0, 255, 0)
C_CLAW         = (0, 200, 255)
C_CLAW_LINE    = (0, 120, 180)
C_WHITE        = (255, 255, 255)
C_WHITE_STALE  = (120, 120, 120)
C_ORANGE       = (0, 165, 255)
C_ORANGE_STALE = (0, 80, 130)
C_LINE         = (200, 200, 0)
C_TEXT         = (255, 255, 255)
C_BG           = (0, 0, 0)


class BallMemory:
    def __init__(self):
        self._balls: list[dict] = []

    def update(self, detections: list[tuple[str, tuple]]):
        for b in self._balls:
            b["fresh"] = False
        for kind, px in detections:
            best_idx, best_dist = None, float("inf")
            for i, b in enumerate(self._balls):
                if b["kind"] != kind:
                    continue
                d = math.hypot(px[0] - b["px"][0], px[1] - b["px"][1])
                if d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx is not None and best_dist < MERGE_RADIUS_PX:
                self._balls[best_idx]["px"]    = px
                self._balls[best_idx]["fresh"] = True
            else:
                self._balls.append({"kind": kind, "px": px, "fresh": True})
                log.debug("New ball added — %s at (%.0f, %.0f)", kind, px[0], px[1])
                print(f"Ball found ({kind}): px=({px[0]:.0f},{px[1]:.0f})")

    @property
    def balls(self):
        return list(self._balls)


def get_robot_and_claw(aruco_detector, frame, homography, w, h):
    """
    Returns (robot_px, robot_angle, claw_px) — all in floor-projected pixel space.

    robot_px    — marker centre corrected to the floor plane
    robot_angle — heading in degrees (recomputed after height correction)
    claw_px     — claw tip floor position, projected from robot_px along robot_angle

    We project from the floor-corrected marker position using the corrected angle.
    In floor space the pixel scale is uniform (PX_PER_CM), so MARKER_TO_CLAW_PX
    corresponds exactly to 16.8 cm regardless of field position or heading.
    No separate claw height correction is needed — we want the floor projection.
    """
    center_raw, forward_raw, angle_raw = detect_robot_pose_in_warped_coords(
        aruco_detector, frame, homography
    )
    if center_raw is None:
        return None, None, None

    robot_px          = correct_robot_height(center_raw,  CAMERA_CENTER_PX, CAMERA_HEIGHT_CM,
                                             ROBOT_MARKER_HEIGHT_CM, w, h, FIELD_WIDTH_CM, FIELD_HEIGHT_CM)
    forward_corrected = correct_robot_height(forward_raw, CAMERA_CENTER_PX, CAMERA_HEIGHT_CM,
                                             ROBOT_MARKER_HEIGHT_CM, w, h, FIELD_WIDTH_CM, FIELD_HEIGHT_CM)

    if robot_px is None or forward_corrected is None:
        return None, None, None

    robot_angle = math.degrees(math.atan2(
        forward_corrected[1] - robot_px[1],
        forward_corrected[0] - robot_px[0],
    ))

    # Project forward in floor space — scale is uniform here
    rad     = math.radians(robot_angle)
    claw_px = (robot_px[0] + MARKER_TO_CLAW_PX * math.cos(rad),
               robot_px[1] + MARKER_TO_CLAW_PX * math.sin(rad))

    return robot_px, robot_angle, claw_px


def draw_arrow(img, origin, angle_deg, length=60, colour=C_ROBOT, thickness=2):
    rad = math.radians(angle_deg)
    ep  = (int(origin[0] + length * math.cos(rad)),
           int(origin[1] + length * math.sin(rad)))
    cv2.arrowedLine(img, (int(origin[0]), int(origin[1])), ep,
                    colour, thickness, tipLength=0.3)


def put_label(img, text, pos, scale=0.55, colour=C_TEXT, bg=C_BG):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + bl), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def main():
    field_model    = load_field_model()
    object_model   = load_object_model()
    aruco_detector = create_detector()
    stream         = open_stream()

    mtx, dist   = load_calibration()
    undist_maps = None
    if mtx is not None:
        undist_maps = build_undistort_maps(mtx, dist, (CAMERA_WIDTH, CAMERA_HEIGHT))
        log.info("Lens calibration loaded")

    last_corners = None
    memory       = BallMemory()

    log.info("Live measurement tool ready — ESC to quit")

    try:
        while True:
            frame = stream.latest()
            if frame is None:
                continue
            frame = undistort_frame(frame, undist_maps)

            last_corners = detect_field(field_model, frame, last_corners)
            if last_corners is None:
                cv2.putText(frame, "Waiting for field...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Measure", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            warped, homography = warp_field(frame, last_corners)
            h, w = warped.shape[:2]

            robot_px, robot_angle, claw_px = get_robot_and_claw(
                aruco_detector, frame, homography, w, h
            )

            # Ball detections → memory
            detections = detect_objects(object_model, warped)
            if robot_px is not None:
                detections = filter_detections_near_robot(detections, robot_px)

            seen = []
            for det in detections:
                name = det.class_name
                if name in ("wb", "ob"):
                    corrected = correct_robot_height(
                        det.center, CAMERA_CENTER_PX, CAMERA_HEIGHT_CM, BALL_HEIGHT_CM,
                        w, h, FIELD_WIDTH_CM, FIELD_HEIGHT_CM,
                    )
                    seen.append((name, corrected))
            memory.update(seen)

            # Draw
            vis = warped.copy()

            for b in memory.balls:
                kind, ball_px, fresh = b["kind"], b["px"], b["fresh"]
                colour = (C_WHITE if fresh else C_WHITE_STALE) if kind == "wb" \
                         else (C_ORANGE if fresh else C_ORANGE_STALE)
                cv2.circle(vis, (int(ball_px[0]), int(ball_px[1])), 8, colour, 2 if fresh else 1)

            if robot_px is not None and claw_px is not None:
                rx, ry = int(robot_px[0]), int(robot_px[1])
                cx, cy = int(claw_px[0]),  int(claw_px[1])

                cv2.circle(vis, (rx, ry), 6, C_ROBOT, -1)
                draw_arrow(vis, robot_px, robot_angle)
                cv2.line(vis, (rx, ry), (cx, cy), C_CLAW_LINE, 1)
                cv2.circle(vis, (cx, cy), 5, C_CLAW, -1)

                put_label(vis,
                          f"Marker=({robot_px[0]:.0f},{robot_px[1]:.0f})  "
                          f"Claw=({claw_px[0]:.0f},{claw_px[1]:.0f})  "
                          f"heading={robot_angle:.1f}°",
                          (8, 20))

                for i, b in enumerate(memory.balls):
                    kind, ball_px, fresh = b["kind"], b["px"], b["fresh"]
                    bx, by = int(ball_px[0]), int(ball_px[1])

                    dist_px = math.hypot(ball_px[0] - claw_px[0],
                                         ball_px[1] - claw_px[1])
                    bearing = angle_to_target(claw_px, ball_px)
                    deg_off = angle_error(robot_angle, bearing)

                    cv2.line(vis, (cx, cy), (bx, by),
                             C_LINE if fresh else (80, 80, 0), 1)

                    stale = "" if fresh else " [stale]"
                    put_label(vis, f"{kind}  {dist_px:.0f}px  {deg_off:+.1f}°{stale}",
                              (bx + 10, by))
                    put_label(vis,
                              f"Ball {i+1} ({kind}){stale}: "
                              f"({ball_px[0]:.0f},{ball_px[1]:.0f})  "
                              f"claw→ball={dist_px:.0f}px  off={deg_off:+.1f}°",
                              (8, 44 + i * 22))

                put_label(vis, f"Known balls: {len(memory.balls)}",
                          (8, 44 + len(memory.balls) * 22))
            else:
                put_label(vis, "No robot detected", (8, 20), colour=(0, 0, 255))

            cv2.imshow("Measure", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
