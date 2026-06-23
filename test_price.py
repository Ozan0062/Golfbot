"""
test_price.py — live camera tool for checking the routing WEIGHT (in motor
rotations) from the robot to every ball found.

Built on test_measure.py, but instead of raw claw->ball px/degrees it shows the
controller.motion.get_price weight for each ball: the rotations the state machine
would actually spend driving there (wall/corner staging + cross dodge + the
turn-to-face), i.e. exactly the edge weight you'd feed a TSP graph.

On the warped top-down view it draws:
  - Robot marker dot + heading arrow
  - The centre cross (what get_price routes around), if detected
  - Every ball, the planned route to it (faint polyline through any staging /
    dodge waypoints), and a label with its weight in rotations
  - The cheapest ball's route highlighted
  - A side panel listing all balls sorted cheapest-first

Run:
    python test_price.py

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
from controller.motion       import get_price, plan_route_waypoints
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_CENTER_PX, CAMERA_HEIGHT_CM,
    ROBOT_MARKER_HEIGHT_CM,
    FIELD_WIDTH_CM, FIELD_HEIGHT_CM,
)

# ── Physical measurements ─────────────────────────────────────────────────────
BALL_HEIGHT_CM  = 0.0    # golf ball parallax correction (negligible)
CROSS_HEIGHT_CM = 0.0    # cross sits on the floor — no parallax correction

MERGE_RADIUS_PX = 40     # max px shift for same-ball detection across frames

setup_logging()
log = get_logger(__name__)

log.info("Price tool ready — weights are get_price() rotations from robot to each ball")

# ── colours ───────────────────────────────────────────────────────────────────
C_ROBOT        = (0, 255, 0)
C_WHITE        = (255, 255, 255)
C_WHITE_STALE  = (120, 120, 120)
C_ORANGE       = (0, 165, 255)
C_ORANGE_STALE = (0, 80, 130)
C_CROSS        = (200, 0, 200)
C_ROUTE        = (170, 170, 170)   # planned route to a ball
C_ROUTE_BEST   = (0, 255, 255)     # cheapest ball's route
C_TEXT         = (255, 255, 255)
C_BG           = (0, 0, 0)


class BallMemory:
    """Merge ball detections across frames so labels don't flicker (from test_measure)."""

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

    @property
    def balls(self):
        return list(self._balls)


def get_robot_pose(aruco_detector, frame, homography, w, h):
    """
    Returns (robot_px, robot_angle) in floor-projected pixel space — the marker
    centre corrected to the floor plane and the heading recomputed from it.

    This is the same rotation centre and heading the state machine drives from,
    so it's the right start point + heading for get_price().
    """
    center_raw, forward_raw, _ = detect_robot_pose_in_warped_coords(
        aruco_detector, frame, homography
    )
    if center_raw is None:
        return None, None

    robot_px          = correct_robot_height(center_raw,  CAMERA_CENTER_PX, CAMERA_HEIGHT_CM,
                                             ROBOT_MARKER_HEIGHT_CM, w, h, FIELD_WIDTH_CM, FIELD_HEIGHT_CM)
    forward_corrected = correct_robot_height(forward_raw, CAMERA_CENTER_PX, CAMERA_HEIGHT_CM,
                                             ROBOT_MARKER_HEIGHT_CM, w, h, FIELD_WIDTH_CM, FIELD_HEIGHT_CM)
    if robot_px is None or forward_corrected is None:
        return None, None

    robot_angle = math.degrees(math.atan2(
        forward_corrected[1] - robot_px[1],
        forward_corrected[0] - robot_px[0],
    ))
    return robot_px, robot_angle


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


def draw_route(img, start_px, waypoints, colour, thickness=1):
    """Draw the planned polyline start_px -> wp1 -> ... -> ball."""
    prev = (int(start_px[0]), int(start_px[1]))
    for wp in waypoints:
        nxt = (int(wp[0]), int(wp[1]))
        cv2.line(img, prev, nxt, colour, thickness, cv2.LINE_AA)
        prev = nxt


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

    log.info("Live price tool ready — ESC to quit")

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
                cv2.imshow("Price", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            warped, homography = warp_field(frame, last_corners)
            h, w = warped.shape[:2]

            robot_px, robot_angle = get_robot_pose(aruco_detector, frame, homography, w, h)

            # Ball + cross detections
            detections = detect_objects(object_model, warped)
            if robot_px is not None:
                detections = filter_detections_near_robot(detections, robot_px)

            seen = []
            cross_px, cross_size, best_cross_conf = None, None, 0.0
            for det in detections:
                name = det["class_name"]
                if name in ("wb", "ob"):
                    corrected = correct_robot_height(
                        det["center"], CAMERA_CENTER_PX, CAMERA_HEIGHT_CM, BALL_HEIGHT_CM,
                        w, h, FIELD_WIDTH_CM, FIELD_HEIGHT_CM,
                    )
                    seen.append((name, corrected))
                elif name == "cross" and det.get("confidence", 1.0) >= best_cross_conf:
                    cross_px        = det["center"]
                    cross_size      = det.get("size")
                    best_cross_conf = det.get("confidence", 1.0)
            memory.update(seen)

            # ── Price every ball with get_price ──────────────────────────────
            priced = []   # (weight, kind, ball_px, fresh, route)
            if robot_px is not None:
                for b in memory.balls:
                    route  = plan_route_waypoints(robot_px, b["px"], cross_px, cross_size)
                    weight = get_price(robot_px, b["px"],
                                       cross_px=cross_px, cross_size_px=cross_size,
                                       start_angle_deg=robot_angle)
                    priced.append((weight, b["kind"], b["px"], b["fresh"], route))
            priced.sort(key=lambda p: p[0])         # cheapest first

            # ── Draw ─────────────────────────────────────────────────────────
            vis = warped.copy()

            if cross_px is not None:
                cx, cy = int(cross_px[0]), int(cross_px[1])
                cv2.drawMarker(vis, (cx, cy), C_CROSS, cv2.MARKER_TILTED_CROSS, 26, 3)
                put_label(vis, "cross", (cx + 12, cy), scale=0.5, colour=C_CROSS)

            if robot_px is None:
                put_label(vis, "No robot detected", (8, 20), colour=(0, 0, 255))
                cv2.imshow("Price", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            # Robot
            rx, ry = int(robot_px[0]), int(robot_px[1])
            cv2.circle(vis, (rx, ry), 6, C_ROBOT, -1)
            draw_arrow(vis, robot_px, robot_angle)
            put_label(vis,
                      f"Robot=({robot_px[0]:.0f},{robot_px[1]:.0f})  heading={robot_angle:.1f}°  "
                      f"cross={'yes' if cross_px else 'no'}",
                      (8, 20))

            # Routes + ball weights (cheapest route highlighted)
            for rank, (weight, kind, ball_px, fresh, route) in enumerate(priced):
                bx, by = int(ball_px[0]), int(ball_px[1])
                is_best = (rank == 0)

                draw_route(vis, robot_px, route,
                           C_ROUTE_BEST if is_best else C_ROUTE,
                           thickness=2 if is_best else 1)

                colour = (C_WHITE if fresh else C_WHITE_STALE) if kind == "wb" \
                         else (C_ORANGE if fresh else C_ORANGE_STALE)
                cv2.circle(vis, (bx, by), 8, colour, 2 if fresh else 1)

                stale = "" if fresh else " [stale]"
                put_label(vis, f"{kind} {weight:.2f} rot{stale}", (bx + 10, by),
                          colour=C_ROUTE_BEST if is_best else C_TEXT)

            # Side panel: balls sorted by weight (cheapest = where the robot would go)
            put_label(vis, f"Weights robot -> ball (rotations) | balls: {len(priced)}", (8, 46))
            for rank, (weight, kind, ball_px, fresh, route) in enumerate(priced):
                stale  = "" if fresh else " [stale]"
                marker = " <= nearest" if rank == 0 else ""
                put_label(vis,
                          f"{rank+1}. {kind}  {weight:6.2f} rot  "
                          f"({ball_px[0]:.0f},{ball_px[1]:.0f}){stale}{marker}",
                          (8, 68 + rank * 22),
                          colour=C_ROUTE_BEST if rank == 0 else C_TEXT)

            cv2.imshow("Price", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
