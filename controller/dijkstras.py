import math
import networkx as nx

from config import GOAL_POSITION_CM, WARPED_WIDTH, WARPED_HEIGHT, FIELD_WIDTH_CM, FIELD_HEIGHT_CM, CORNER_STAGE_DISTANCES_PX, WALL_MARGIN_PX, FIELD_EDGE_MARGIN_PX
from controller.motion import angle_to_rotations, px_to_rotations
from controller.navigation import angle_to_target, cm_to_pixels, px_to_cm, classify_zone, wall_approach_angle, staging_point
from vision.tracker import WorldState

WALL_BALL_PENALTY = 0
CORNER_BALL_PENALTY = 0
CROSS_BLOCK_PENALTY = 6
CROSS_CLEARANCE_CM = 0

def angle_rotations(robot_pos_cm, target_pos_cm):
    angle = angle_to_target(robot_pos_cm, target_pos_cm)
    return angle_to_rotations(angle)

def line_intersects_obstacle(robot_coords, ball_coords, cross_coords, clearance=15.0) -> bool:
    """Check if the line between robot and ball intersects the cross clearance zone."""
    ax, ay = robot_coords
    bx, by = ball_coords
    ox, oy = cross_coords
    
    ab_x, ab_y = bx - ax, by - ay
    ao_x, ao_y = ox - ax, oy - ay
    ab_len_sq = ab_x**2 + ab_y**2
    
    if ab_len_sq == 0:
        return math.dist(robot_coords, cross_coords) < clearance
        
    t = (ao_x * ab_x + ao_y * ab_y) / ab_len_sq
    
    if t <= 0.0:
        return False
        
    t = min(1.0, t)
    
    proj_x = ax + t * ab_x
    proj_y = ay + t * ab_y
    
    return math.dist((ox, oy), (proj_x, proj_y)) < clearance

def _add_staging_nodes(G, ball_id, ball_px, robot_px):
    """Create a chain of staging nodes leading to a wall/corner ball."""
    _, walls = classify_zone(ball_px, WALL_MARGIN_PX, WARPED_WIDTH, WARPED_HEIGHT)
    angle = wall_approach_angle(walls) if walls else None
    
    if angle is None:
        return
    
    prev_stg_id = None
    for i, dist in enumerate(CORNER_STAGE_DISTANCES_PX):
        sp = staging_point(ball_px, angle, dist)
        sp = (max(FIELD_EDGE_MARGIN_PX, min(sp[0], WARPED_WIDTH - FIELD_EDGE_MARGIN_PX)),
              max(FIELD_EDGE_MARGIN_PX, min(sp[1], WARPED_HEIGHT - FIELD_EDGE_MARGIN_PX)))
        
        stg_id = f"stg_{ball_id}_{i}"
        
        G.add_node(stg_id, pos=px_to_cm(sp), pos_px=sp,
                   type="staging", penalty=0.0, parent_ball=ball_id)
        
        if prev_stg_id is None:
            if robot_px:
                dist_rot = px_to_rotations(math.dist(robot_px, sp))
                G.add_edge("robot", stg_id, weight=dist_rot, distance=dist_rot)
        else:
            prev_sp = G.nodes[prev_stg_id]["pos_px"]
            dist_rot = px_to_rotations(math.dist(prev_sp, sp))
            G.add_edge(prev_stg_id, stg_id, weight=dist_rot, distance=dist_rot)
            
        prev_stg_id = stg_id
        
    # Connect the final staging point to the target ball
    if prev_stg_id is not None:
        dist_rot = px_to_rotations(math.dist(sp, ball_px))
        G.add_edge(prev_stg_id, ball_id, weight=dist_rot, distance=dist_rot)

def create_nodes_and_edges(world: WorldState) -> nx.DiGraph:
    G = nx.DiGraph()
    from config import GOAL_POSITION_PX
    
    def get_cross_penalty(target_pos):
        if world.robot and world.cross:
            if line_intersects_obstacle(world.robot, target_pos, world.cross, CROSS_CLEARANCE_CM):
                return CROSS_BLOCK_PENALTY
        return 0.0
    
    if world.robot and world.robot_px:
        G.add_node("robot", pos=world.robot, pos_px=world.robot_px, type="robot", penalty=0.0)
    
    ball_idx = 0
    
    for ball_cm, ball_px in zip(world.white_balls, world.white_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        penalty = get_cross_penalty(pos_cm) + angle_rotations(world.robot, pos_cm)
        G.add_node(f"wb_{ball_idx}", pos=pos_cm, pos_px=pos_px, type="open", penalty=penalty)
        ball_idx += 1
        
    for ball_cm, ball_px in zip(world.white_wall_balls, world.white_wall_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        penalty = WALL_BALL_PENALTY + get_cross_penalty(pos_cm) + angle_rotations(world.robot, pos_cm)
        bid = f"wb_{ball_idx}"
        G.add_node(bid, pos=pos_cm, pos_px=pos_px, type="wall", penalty=penalty)
        _add_staging_nodes(G, bid, pos_px, world.robot_px)
        ball_idx += 1
        
    for ball_cm, ball_px in zip(world.white_corner_balls, world.white_corner_balls_px):
        pos_cm = (ball_cm[0], ball_cm[1])
        pos_px = (ball_px[0], ball_px[1])
        penalty = CORNER_BALL_PENALTY + get_cross_penalty(pos_cm) + angle_rotations(world.robot, pos_cm)
        bid = f"wb_{ball_idx}"
        G.add_node(bid, pos=pos_cm, pos_px=pos_px, type="corner", penalty=penalty)
        _add_staging_nodes(G, bid, pos_px, world.robot_px)
        ball_idx += 1
        
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
    
    nodes = list(G.nodes(data=True))
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
                
            name_a, data_a = nodes[i]
            name_b, data_b = nodes[j]
            
            # Prevent outgoing edges from staging nodes
            if data_a["type"] == "staging":
                continue
            
            # Prevent direct incoming edges to wall/corner balls (must use staging chain)
            if data_b["type"] in ("wall", "corner"):
                continue
            
            dist_px = math.dist(data_a["pos_px"], data_b["pos_px"])
            dist_rotations = px_to_rotations(dist_px)
            weight = dist_rotations + data_b["penalty"]
            
            G.add_edge(name_a, name_b, weight=weight, distance=dist_rotations)
            
    return G

def calculate_best_route(G: nx.DiGraph) -> list[dict]:
    if not G.has_node("robot"):
        return []
        
    try:
        lengths = nx.single_source_dijkstra_path_length(G, source="robot", weight="weight")
    except nx.NodeNotFound:
        return []
        
    # Order targets by Dijkstra distance
    sorted_whites = [node for node in lengths if str(node).startswith("wb_")]
    unreachable_whites = [n for n in G.nodes() if n.startswith("wb_") and n not in lengths]
    sorted_whites.extend(unreachable_whites)
    
    path_nodes = sorted_whites
    if G.has_node("ob"):
        path_nodes.append("ob")
    if G.has_node("goal"):
        path_nodes.append("goal")
        
    result = []
    robot_px = G.nodes["robot"]["pos_px"]
    
    for target_ball in path_nodes:
        try:
            p = nx.shortest_path(G, source="robot", target=target_ball, weight="weight")
            
            for step in p[1:]:
                # Skip target nodes if robot is physically close to avoid infinite loops
                if math.dist(robot_px, G.nodes[step]["pos_px"]) < 10.0:
                    continue
                    
                step_data = G.nodes[step]
                result.append({
                    "id": step,
                    "pos_cm": step_data["pos"],
                    "pos_px": step_data.get("pos_px", (0, 0)),
                    "type": step_data["type"],
                    "parent_ball": target_ball if step_data["type"] == "staging" else None
                })
                
        except nx.NetworkXNoPath:
            node_data = G.nodes[target_ball]
            result.append({
                "id": target_ball,
                "pos_cm": node_data["pos"],
                "pos_px": node_data.get("pos_px", (0, 0)),
                "type": node_data["type"]
            })
            
    return result

def get_path(world: WorldState):
    graph = create_nodes_and_edges(world)
    return calculate_best_route(graph)