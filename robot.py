from sensors import read_distance
from navigation import decide_action
from motors import execute_action

class Robot:

    def update(self):

        distance = read_distance()

        action = decide_action(distance)

        execute_action(action)