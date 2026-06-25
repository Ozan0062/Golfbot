"""
Every state in the state machine returns one of these.
The overlay, logging, and ev3_controller all speak this enum.
"""

from enum import Enum, auto


class Command(Enum):
    STOP      = auto()
    FORWARD   = auto()
    BACKWARD  = auto()
    LEFT      = auto()
    RIGHT     = auto()
    COLLECT   = auto()
    RELEASE   = auto()
