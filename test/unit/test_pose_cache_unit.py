"""
test_pose_cache_unit.py - offline unit tests for cached ArUco pose handling.
Mocks time so timeout and settle-window behaviour is deterministic.
"""

import unittest
from unittest.mock import patch

from controller.pose_cache import POSE_TIMEOUT_S, SETTLE_S, PoseCache
from test.world_state_helpers import world_state


class PoseCacheUnitTests(unittest.TestCase):
    def test_update_returns_fresh_pose_from_world_dict(self):
        cache = PoseCache()
        world = world_state(robot=(10.0, 20.0), robot_px=(100, 200), robot_angle=45.0)

        with patch("builtins.print"), patch("controller.pose_cache.time.time", return_value=1.0):
            pose = cache.update(world)

        self.assertEqual(pose.pos, (10.0, 20.0))
        self.assertEqual(pose.px, (100, 200))
        self.assertEqual(pose.angle, 45.0)

    def test_cached_pose_is_returned_until_timeout(self):
        cache = PoseCache()
        world = world_state(robot=(10.0, 20.0), robot_px=(100, 200), robot_angle=45.0)

        with patch("builtins.print"), patch("controller.pose_cache.time.time", return_value=1.0):
            cache.update(world)

        with patch(
            "controller.pose_cache.time.time",
            return_value=1.0 + POSE_TIMEOUT_S / 2,
        ):
            pose = cache.update(world_state(robot=None, robot_angle=None))

        self.assertIsNotNone(pose)
        self.assertEqual(pose.pos, (10.0, 20.0))

    def test_cached_pose_expires_after_timeout(self):
        cache = PoseCache()
        world = world_state(robot=(10.0, 20.0), robot_px=(100, 200), robot_angle=45.0)

        with patch("builtins.print"), patch("controller.pose_cache.time.time", return_value=1.0):
            cache.update(world)

        with patch(
            "controller.pose_cache.time.time",
            return_value=1.0 + POSE_TIMEOUT_S + 0.1,
        ):
            pose = cache.update(world_state(robot=None, robot_angle=None))

        self.assertIsNone(pose)

    def test_invalidate_blocks_pose_updates_during_settle_window(self):
        cache = PoseCache()

        with patch("controller.pose_cache.time.time", return_value=10.0):
            cache.invalidate()

        with patch("builtins.print"), patch(
            "controller.pose_cache.time.time",
            return_value=10.0 + SETTLE_S / 2,
        ):
            pose = cache.update(
                world_state(robot=(10.0, 20.0), robot_px=(100, 200), robot_angle=45.0)
            )

        self.assertIsNone(pose)


if __name__ == "__main__":
    unittest.main()

