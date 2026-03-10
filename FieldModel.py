"""
FIELD MODEL

Represents the layout of the golf field.

Tracks important elements such as:
- robot position
- ball position
- goal location
"""


class FieldModel:

    def __init__(self):
        self.robot_position = None
        self.ball_position = None
        self.goal_position = None

    def update_ball(self, position):
        self.ball_position = position

    def update_robot(self, position):
        self.robot_position = position