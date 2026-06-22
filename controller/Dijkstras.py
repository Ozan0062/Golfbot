import math
import networkx as nx

from config import GOAL_POSITION_CM, WARPED_WIDTH, WARPED_HEIGHT, FIELD_WIDTH_CM, FIELD_HEIGHT_CM
from controller.motion import angle_to_rotations, px_to_rotations
from controller.navigation import angle_to_target, cm_to_pixels
from vision.tracker import WorldState

WALL_BALL_PENALTY = 5      # Added weight/cost for balls near walls
CORNER_BALL_PENALTY = 10    # Added weight/cost for balls in corners
CROSS_BLOCK_PENALTY = 6   # Ekstra straf for bolde bag krydset (fra robottens perspektiv)
CROSS_CLEARANCE_CM = 15    # Afstand fra kryds

def angle_rotations(robot_pos_cm,target_pos_cm):
    angle = angle_to_target(robot_pos_cm, target_pos_cm)
    rotations = angle_to_rotations(angle)
    return rotations

def line_intersects_obstacle(robot_coords, ball_coords, cross_coords, clearance=15.0) -> bool:
    """Cheks if robot and ball intersects with cross"""
    ax, ay = robot_coords
    bx, by = ball_coords
    ox, oy = cross_coords
    
    ab_x, ab_y = bx - ax, by - ay
    ao_x, ao_y = ox - ax, oy - ay
    ab_len_sq = ab_x**2 + ab_y**2
    
    if ab_len_sq == 0:
        return math.dist(robot_coords, cross_coords) < clearance
        
    t = (ao_x * ab_x + ao_y * ab_y) / ab_len_sq
    
    # Hvis t <= 0, bevæger robotten sig væk fra krydset (krydset ligger bagved eller til siden)
    # Dermed er krydset ikke i vejen for den direkte rute!
    if t <= 0.0:
        return False
        
    # Begræns t til 1.0. Hvis t > 1 ligger forhindringen længere væk end selve bolden.
    # Ved at klippe til 1.0 sikrer vi, at hvis BOLDEN ligger inden for radius, får den stadig straf.
    t = min(1.0, t)
    
    proj_x = ax + t * ab_x
    proj_y = ay + t * ab_y
    
    return math.dist((ox, oy), (proj_x, proj_y)) < clearance

def create_nodes_and_edges(world: WorldState) -> nx.DiGraph:
    G = nx.DiGraph()
    
    # Helper: Tjekker om bolden er bag krydset set fra robotten
    def get_cross_penalty(target_pos):
        if world.robot and world.cross:
            if line_intersects_obstacle(world.robot, target_pos, world.cross, CROSS_CLEARANCE_CM):
                return CROSS_BLOCK_PENALTY
        return 0.0
    
    from config import GOAL_POSITION_PX
    
    # 1. Add Robot node
    if world.robot and world.robot_px:
        G.add_node("robot", pos=world.robot, pos_px=world.robot_px, type="robot", penalty=0.0)
    
    ball_idx = 0
    
    # 2. Add open field white balls
    for ball_cm, ball_px in zip(world.white_balls, world.white_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        penalty = get_cross_penalty(pos_cm) + angle_rotations(world.robot, pos_cm)
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="open", penalty=penalty)
        ball_idx += 1
        
    # 3. Add wall white balls
    for ball_cm, ball_px in zip(world.white_wall_balls, world.white_wall_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        penalty = WALL_BALL_PENALTY + get_cross_penalty(pos_cm) + angle_rotations(world.robot, pos_cm)
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="wall", penalty=penalty)
        ball_idx += 1
        
    # 4. Add corner white balls
    for ball_cm, ball_px in zip(world.white_corner_balls, world.white_corner_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        penalty = CORNER_BALL_PENALTY + get_cross_penalty(pos_cm) + angle_rotations(world.robot, pos_cm)
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="corner", penalty=penalty)
        ball_idx += 1
        
    # 5. Add orange ball
    if world.ob and world.ob_px:
        pos_cm = (world.ob[0], world.ob[1])
        pos_px = (world.ob_px[0], world.ob_px[1])
        zone = world.ob[2] if len(world.ob) > 2 else "open"
        penalty = get_cross_penalty(pos_cm)
        if zone == "wall":
            penalty += WALL_BALL_PENALTY
        elif zone == "corner":
            penalty += CORNER_BALL_PENALTY
        G.add_node("ob", pos=pos_cm, pos_px=pos_px, type=zone, penalty=penalty)
        
    G.add_node("goal", pos=GOAL_POSITION_CM, pos_px=GOAL_POSITION_PX, type="goal", penalty=0.0)
    
    # 6. Create edges between all nodes
    nodes = list(G.nodes(data=True))
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
                
            name_a, data_a = nodes[i]
            name_b, data_b = nodes[j]
            
            # Udregn afstanden i pixels
            dist_px = math.dist(data_a["pos_px"], data_b["pos_px"])
            
            # Omdan pixel-afstanden til motoromdrejninger
            dist_rotations = px_to_rotations(dist_px)
            
            penalty = data_b["penalty"]
            
            weight = dist_rotations + penalty
            
            # Vi gemmer dist_rotations, men også den gamle dist i grafen, just in case
            G.add_edge(name_a, name_b, weight=weight, distance=dist_rotations)
            
    return G

def calculate_best_route(G: nx.DiGraph) -> list[dict]:
    if not G.has_node("robot"):
        return []
        
    try:
        lengths = nx.single_source_dijkstra_path_length(G, source="robot", weight="weight")
    except nx.NodeNotFound:
        return []
        
    # Find alle hvide bolde og sorter dem efter Dijkstra-afstanden.
    # get(n, inf) sikrer at bolde, der potentielt er afskåret, bliver lagt bagerst.
    white_balls = [n for n in G.nodes() if n.startswith("wb_")]
    sorted_whites = sorted(white_balls, key=lambda n: lengths.get(n, float('inf')))
    
    path_nodes = sorted_whites
    
    # Always add the orange ball last
    if G.has_node("ob"):
        path_nodes.append("ob")
        
    # Always add the goal as the absolute final destination
    if G.has_node("goal"):
        path_nodes.append("goal")
        
    # Format output with positions and types
    result = []
    for node_id in path_nodes:
        node_data = G.nodes[node_id]
        pos_cm = node_data["pos"]
        # Hent pos_px direkte fra noden! (Den ægte YOLO-pixel)
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
 