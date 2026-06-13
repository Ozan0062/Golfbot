import sys
import types
import unittest
from unittest.mock import patch

from controller.commands import Command
from controller.navigation import cm_to_pixels, safe_approach_point

# Keep these controller tests independent of the Christofides dependency.
tsp_module = types.ModuleType("controller.tsp_christofides")
tsp_module.christofides_route = lambda points: list(range(len(points)))
sys.modules["controller.tsp_christofides"] = tsp_module

from controller.state_machine import GolfBotController, MAX_DRIVE_PX
from config import (
    FIELD_HEIGHT_CM,
    FIELD_WIDTH_CM,
    PIXELS_PER_ROTATION,
    WARPED_HEIGHT,
    WARPED_WIDTH,
)


def make_world(robot, robot_px, angle, ball):
    return {
        "robot": robot,
        "robot_px": robot_px,
        "robot_angle": angle,
        "white_balls": [ball],
        "white_balls_px": [
            cm_to_pixels(
                ball,
                WARPED_WIDTH,
                WARPED_HEIGHT,
                FIELD_WIDTH_CM,
                FIELD_HEIGHT_CM,
            )
        ],
        "ob": None,
        "ob_px": None,
    }


class CornerSafetyTests(unittest.TestCase):

    def test_corner_target_is_moved_inside_safety_zone(self):
        self.assertEqual(
            safe_approach_point((0, 0), 15, FIELD_WIDTH_CM, FIELD_HEIGHT_CM),
            (15, 15),
        )
        self.assertEqual(
            safe_approach_point((180, 120), 15, FIELD_WIDTH_CM, FIELD_HEIGHT_CM),
            (165, 105),
        )

    @patch("controller.state_machine.robot.drive")
    def test_drive_toward_corner_is_limited_to_short_step(self, drive):
        controller = GolfBotController()

        command = controller.update(
            make_world((90, 60), (320, 240), -146.3, (0, 0))
        )

        self.assertEqual(command, Command.FORWARD)
        drive.assert_called_once()
        rotations = drive.call_args.args[0]
        self.assertAlmostEqual(rotations, MAX_DRIVE_PX / PIXELS_PER_ROTATION)

    @patch("controller.state_machine.robot.collect")
    def test_robot_collects_from_safe_approach_point(self, collect):
        controller = GolfBotController()

        command = controller.update(
            make_world((15, 15), (53.333, 60), -135, (0, 0))
        )

        self.assertEqual(command, Command.COLLECT)
        collect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
