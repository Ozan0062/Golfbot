import math
import networkx as nx

from config import CAMERA_CENTER_PX, GOAL_POSITION_CM
from controller.motion import get_price
from vision.tracker import WorldState

def create_nodes_and_edges(world: WorldState, ignored_px=None) -> nx.DiGraph:
    G = nx.DiGraph()
    
    from config import GOAL_POSITION_PX
    ignored_px = ignored_px or []

    def is_ignored(pos_px):
        if pos_px is None:
            return False
        return any(math.dist((pos_px[0], pos_px[1]), ignored) < 12 for ignored in ignored_px)
    
    # 1. Add Robot node
    if world.robot and world.robot_px:
        G.add_node("robot", pos=world.robot, pos_px=world.robot_px, type="robot", penalty=0.0)
    
    ball_idx = 0
    
    # 2. Add open field white balls
    for ball_cm, ball_px in zip(world.white_balls, world.white_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        if is_ignored(pos_px):
            continue
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="open", penalty=0)
        ball_idx += 1
        
    # 3. Add wall white balls
    for ball_cm, ball_px in zip(world.white_wall_balls, world.white_wall_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        if is_ignored(pos_px):
            continue
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="wall", penalty=0)
        ball_idx += 1
        
    # 4. Add corner white balls
    for ball_cm, ball_px in zip(world.white_corner_balls, world.white_corner_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        if is_ignored(pos_px):
            continue
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="corner", penalty=0)
        ball_idx += 1
        
    # 5. Add orange ball
    if world.ob and world.ob_px:
        pos_cm = (world.ob[0], world.ob[1])
        pos_px = (world.ob_px[0], world.ob_px[1])
        if not is_ignored(pos_px):
            zone = world.ob[2] if len(world.ob) > 2 else "open"
            G.add_node("ob", pos=pos_cm, pos_px=pos_px, type=zone, penalty=0)
        
    G.add_node("goal", pos=GOAL_POSITION_CM, pos_px=GOAL_POSITION_PX, type="goal", penalty=0.0)
    
    # 6. Create edges ONLY from the robot to all other nodes (for Nearest Neighbor)
    if not G.has_node("robot"):
        return G
        
    robot_data = G.nodes["robot"]
    nodes = list(G.nodes(data=True))
    
    for name_b, data_b in nodes:
        if name_b == "robot" or name_b == "goal":
            continue
            
        try:
            weight = get_price(
                robot_data["pos_px"],
                data_b["pos_px"],
                cross_px=world.cross_px,
                cross_size_px=70 * 70,
                start_angle_deg=world.robot_angle,
            )
        except Exception:
            weight = float('inf')
            
        G.add_edge("robot", name_b, weight=weight)
        
    return G

def find_nearest(world: WorldState, ignored_px=None) -> dict | None:
    G = create_nodes_and_edges(world, ignored_px)
    
    if not G.has_node("robot"):
        return None
        
    # Find all white balls directly connected to the robot
    white_balls = []
    for neighbor in G.successors("robot"):
        if str(neighbor).startswith("wb_"):
            weight = G["robot"][neighbor].get("weight", float('inf'))
            white_balls.append((neighbor, weight))
            
    # If there are white balls, return the nearest
    if white_balls:
        white_balls.sort(key=lambda x: x[1])
        node_id = white_balls[0][0]
    elif G.has_node("ob"):
        # If no white balls, go for orange ball
        node_id = "ob"
    else:
        return None
        
    node_data = G.nodes[node_id]
    return {
        "id": node_id,
        "pos_cm": node_data["pos"],
        "pos_px": node_data.get("pos_px", (0, 0)),
        "type": node_data["type"]
    }
 
