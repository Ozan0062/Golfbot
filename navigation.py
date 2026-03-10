def decide_action(distance):

    if distance < 200:
        return "AVOID"

    else:
        return "SEARCH"