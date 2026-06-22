import math
import networkx as nx

from vision.tracker import WorldState

# Top-level variables for penalties so they can be easily adjusted
WALL_BALL_PENALTY = 50.0       # Added weight/cost for balls near walls
CORNER_BALL_PENALTY = 100.0    # Added weight/cost for balls in corners
CROSS_PENALTY = 999999.0       # Extreme weight so the cross is avoided

def createPath(world: WorldState) -> nx.DiGraph:
    """
    Creates a Directed Graph of all objects on the field.
    The edge weights are calculated as: 
        Euclidean Distance + Penalty of the target node.
    """
    G = nx.DiGraph()
    
    # 1. Add Robot node
    if world.robot:
        G.add_node("robot", pos=world.robot, type="robot", penalty=0.0)
    
    ball_idx = 0
    
    # 2. Add open field white balls
    for ball in world.white_balls:
        pos_cm = (ball[0], ball[1])
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, type="open", penalty=0.0)
        ball_idx += 1
        
    # 3. Add wall white balls
    for ball in world.white_wall_balls:
        pos_cm = (ball[0], ball[1])
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, type="wall", penalty=WALL_BALL_PENALTY)
        ball_idx += 1
        
    # 4. Add corner white balls
    for ball in world.white_corner_balls:
        pos_cm = (ball[0], ball[1])
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, type="corner", penalty=CORNER_BALL_PENALTY)
        ball_idx += 1
        
    # 5. Add orange ball
    if world.ob:
        pos_cm = (world.ob[0], world.ob[1])
        zone = world.ob[2] if len(world.ob) > 2 else "open"
        penalty = 0.0
        if zone == "wall":
            penalty = WALL_BALL_PENALTY
        elif zone == "corner":
            penalty = CORNER_BALL_PENALTY
        G.add_node("ob", pos=pos_cm, type=zone, penalty=penalty)
        
    # 6. Add Cross
    if world.cross:
        pos_cm = (world.cross[0], world.cross[1])
        G.add_node("cross", pos=pos_cm, type="obstacle", penalty=CROSS_PENALTY)
        
    # 7. Create edges between all nodes
    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
                
            name_a, data_a = nodes[i]
            name_b, data_b = nodes[j]
            
            # Distance from A to B
            dist = math.dist(data_a["pos"], data_b["pos"])
            
            # Weight = Distance + Target Node's Penalty
            weight = dist + data_b["penalty"]
            
            G.add_edge(name_a, name_b, weight=weight, distance=dist)
            
    return G