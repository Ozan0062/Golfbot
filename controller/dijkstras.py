import math
import networkx as nx

from config import CAMERA_CENTER_PX, GOAL_POSITION_CM
from controller.motion import get_price
from vision.tracker import WorldState

def create_nodes_and_edges(world: WorldState) -> nx.DiGraph:
    G = nx.DiGraph()
    
    from config import GOAL_POSITION_PX
    
    # 1. Add Robot node
    if world.robot and world.robot_px:
        G.add_node("robot", pos=world.robot, pos_px=world.robot_px, type="robot", penalty=0.0)
    
    ball_idx = 0
    
    # 2. Add open field white balls
    for ball_cm, ball_px in zip(world.white_balls, world.white_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="open", penalty=0)
        ball_idx += 1
        
    # 3. Add wall white balls
    for ball_cm, ball_px in zip(world.white_wall_balls, world.white_wall_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="wall", penalty=0)
        ball_idx += 1
        
    # 4. Add corner white balls
    for ball_cm, ball_px in zip(world.white_corner_balls, world.white_corner_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="corner", penalty=0)
        ball_idx += 1
        
    # 5. Add orange ball
    if world.ob and world.ob_px:
        pos_cm = (world.ob[0], world.ob[1])
        pos_px = (world.ob_px[0], world.ob_px[1])
        zone = world.ob[2] if len(world.ob) > 2 else "open"
        G.add_node("ob", pos=pos_cm, pos_px=pos_px, type=zone, penalty=0)
        
    G.add_node("goal", pos=GOAL_POSITION_CM, pos_px=GOAL_POSITION_PX, type="goal", penalty=0.0)
    
    # 6. Create edges between all nodes
    nodes = list(G.nodes(data=True))
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
                
            name_a, data_a = nodes[i]
            name_b, data_b = nodes[j]
            
            weight = get_price(
                data_a["pos_px"],
                data_b["pos_px"],
                cross_px=world.cross_px,
                cross_size_px=70 * 70,
                start_angle_deg=world.robot_angle,
            )
            
            G.add_edge(name_a, name_b, weight=weight)
            
    return G

def calculate_best_route(G: nx.DiGraph) -> list[dict]:
    if not G.has_node("robot"):
        return []
        
    try:
        lengths = nx.single_source_dijkstra_path_length(G, source="robot", weight="weight")
    except nx.NodeNotFound:
        return []
        
    # Extract reachable white balls directly in the order Dijkstra discovered them
    sorted_whites = [node for node in lengths if str(node).startswith("wb_")]
    
    # Append any unreachable white balls at the end, just in case
    unreachable_whites = [n for n in G.nodes() if n.startswith("wb_") and n not in lengths]
    sorted_whites.extend(unreachable_whites)
    
    path_nodes = sorted_whites
    
    # Always add the orange ball last
    if G.has_node("ob"):
        path_nodes.append("ob")
        
        
    # Format output with positions and types
    result = []
    for node_id in path_nodes:
        node_data = G.nodes[node_id]
        pos_cm = node_data["pos"]
        pos_px = node_data.get("pos_px", (0, 0))
        
        result.append({
            "id": node_id,
            "pos_cm": pos_cm,
            "pos_px": pos_px,
            "type": node_data["type"]
        })
        
    return result

def get_path(world: WorldState):
    graph = create_nodes_and_edges(world)
    path  =  calculate_best_route(graph)
    return path
 