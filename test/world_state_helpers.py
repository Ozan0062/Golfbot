"""
world_state_helpers.py - small offline fixtures for tests that need WorldState.
Keeps tests focused on behaviour instead of repeating empty field defaults.
"""

from vision.tracker import WorldState


def world_state(**overrides):
    world = WorldState()
    for key, value in overrides.items():
        setattr(world, key, value)
    return world
