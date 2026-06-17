import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enum import Enum, auto

from controller.calibration_manager import CalibrationManager
from controller.commands import Command
from controller.pose_cache import PoseCache
from controller.route_manager import RouteManager
from controller import state_machine

class State(Enum):
    SEEK           = auto()   # acquire next target from TSP route
    AVOID          = auto()   # drive to waypoint to get around cross
    ALIGN          = auto()   # turn to face the locked target
    APPROACH       = auto()   # drive toward the locked target
    REVERSE_WHITE  = auto()   # back up, scan for white or orange balls
    REVERSE_ORANGE = auto()   # back up, scan for orange ball only
    DRIVE_GOAL     = auto()   # navigate to goal zone (turn + drive)
    RELEASE        = auto()   # dump balls at goal
    DONE           = auto()   # mission complete

class StateDetector:
    def __init__(self):
        self.state           = State.SEEK
        self._pose           = PoseCache()
        self._route          = RouteManager()
        self._cal            = CalibrationManager()
        self._has_reversed   = False
        self._locked_target  = None   # type RouteTarget None
        self._avoid_target   = None   # type tuple
        

    def update(self, world: dict) -> Command:
        """Run one tick of the state machine. Returns the command executed."""
        pose = self._pose.update(world)
        if pose is None:
            print("[FSM] ArUco not detected -- waiting.")
            return Command.STOP

        self._cal.consume(pose.px, pose.angle)

        if self.state == State.SEEK:
            return self._seek(pose, world)
        if self.state == State.ALIGN:
            return self._align(pose)
        return Command.STOP   # DONE
    
    
    
    def _seek(self, pose, world) -> Command:
        self._locked_target = self._route.get_target(pose.pos, pose.px, world)

        self._transition(State.ALIGN)
        return Command.STOP


    def _align(self, pose) -> Command:
        target = self._locked_target
        if target is None:
            self._transition(State.SEEK)
            return Command.STOP

        heading_error = self._heading_error_to(pose, target.cm)
        print(f"[ALIGN] Heading_Error={heading_error}")

        if abs(heading_error) <= state_machine.ALIGN_THRESHOLD_DEG:
            self._transition(State.APPROACH)
            return Command.STOP

        rotations = state_machine._angle_to_rotations(heading_error)
        if rotations < state_machine.MIN_TURN_ROTATIONS:
            self._transition(State.APPROACH)
            return Command.STOP

        direction = Command.RIGHT if heading_error > 0 else Command.LEFT
        self._execute_turn(pose, rotations, direction)
        return direction

    def _transition(self, new_state):
        print(f"[FSM] Transition: {self.state.name} -> {new_state.name}")
        self.state = new_state

    def _heading_error_to(self, pose, target_cm):
        from controller.navigation import angle_to_target, angle_error
        target_angle = angle_to_target(pose.pos, target_cm)
        return angle_error(pose.angle, target_angle)

    def _execute_turn(self, pose, rotations, direction):
        # Stub for visual testing
        print(f"[TURN] {direction.name} by {rotations:.2f} rot")