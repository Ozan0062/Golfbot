import sys
import os
import math
import matplotlib.pyplot as plt
import networkx as nx

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.tracker import WorldState
from controller.nearest import create_nodes_and_edges, find_nearest

def draw_state(ax, world):
    """Draws current state and return calculated route."""
    ax.clear()
    
    # 1. build graph and calc nearest target
    G = create_nodes_and_edges(world)
    target = find_nearest(world)

    # 2. plot setup
    ax.set_title("Nearest / NetworkX - dynamic simulation (Close window to stop)")
    ax.set_xlim(-10, 190) # Field width + margin
    ax.set_ylim(130, -10) # Field height + margin (inverted Y)
    
    # draw track edges (walls)
    ax.plot([0, 180, 180, 0, 0], [0, 0, 120, 120, 0], color="black", linewidth=5, zorder=1, label="wall/track-edge")
    
    # 3. Draw nodes
    pos = nx.get_node_attributes(G, 'pos')
    
    color_map = {
        "robot": "black",
        "open": "lightgray",
        "wall": "blue",
        "corner": "red",
        "obstacle": "purple",
        "goal": "green"
    }
    
    drawn_labels = set()
    for node, data in G.nodes(data=True):
        c = color_map.get(data["type"], "green")
        if node == "ob": c = "orange"
        
        label = data["type"]
        if label in drawn_labels:
            label = ""
        else:
            drawn_labels.add(label)
            
        ax.scatter(data["pos"][0], data["pos"][1], c=c, s=200, zorder=5, label=label, edgecolors="black")
        ax.text(data["pos"][0], data["pos"][1] - 4, node, fontsize=9, ha="center")
        
        if data["penalty"] > 0 and node != "cross":
            ax.text(data["pos"][0], data["pos"][1] + 6, f"+{data['penalty']} penalty", fontsize=8, color="red", ha="center")

    # draw cross manualy (since its no longer a node)
    if world.cross:
        ax.scatter(world.cross[0], world.cross[1], c="purple", s=200, zorder=5, label="obstacle", edgecolors="black")
        ax.text(world.cross[0], world.cross[1] - 4, "cross", fontsize=9, ha="center")

    # draw robot dir/angle
    if world.robot and world.robot_angle is not None:
        rx, ry = world.robot
        dx = 12 * math.cos(math.radians(world.robot_angle))
        dy = 12 * math.sin(math.radians(world.robot_angle))
        ax.arrow(rx, ry, dx, dy, head_width=3, head_length=4, fc='black', ec='black', zorder=6, width=0.5)

    # 4. draw arrows in sequece
    prev_pos = pos.get("robot", None)
    pos_px_dict = nx.get_node_attributes(G, 'pos_px')
    prev_pos_px = pos_px_dict.get("robot", None)
    
    
    if prev_pos and prev_pos_px and target:
        target_pos = target.get("pos", target.get("pos_cm"))
        target_node_id = target["id"]
        
        arrow_color = plt.cm.Greys(0.8)
        
        ax.annotate("",
                    xy=target_pos, xycoords='data',
                    xytext=prev_pos, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=arrow_color, alpha=0.9, lw=3, shrinkA=12, shrinkB=12))
                                    
        # get acutal edge weight from graph
        edge_weight = G["robot"][target_node_id]["weight"] if G.has_edge("robot", target_node_id) else 0.0
        
        mid_x = (prev_pos[0] + target_pos[0]) / 2
        mid_y = (prev_pos[1] + target_pos[1]) / 2
        ax.text(mid_x, mid_y, f"{edge_weight:.1f} rot", color="darkred", fontsize=8, fontweight="bold", 
                ha="center", va="center", bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    # 5. final gfx setup
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    return target

def main():
    # define our test track
    world = WorldState(
        robot=(20.0, 60.0), # robot starts left side
        white_balls=[
            (90.0, 30.0, "open"),   # above cross
            (90.0, 90.0, "open"),   # under cross
            (60.0, 60.0, "open"),   # left of cross
            (130.0, 60.0, "open")   # right of cross
        ],
        white_wall_balls=[
            (10.0, 60.0, "wall"),   # left wall
            (170.0, 60.0, "wall")   # right wall
        ],
        white_corner_balls=[
            (10.0, 10.0, "corner"),   # top left corner
            (170.0, 110.0, "corner")  # bottom right corner
        ],
        ob=(40.0, 40.0, "open"),
        cross=(90.0, 60.0) # cross placed middle of track!
    )

    from config import WARPED_WIDTH, WARPED_HEIGHT, FIELD_WIDTH_CM, FIELD_HEIGHT_CM
    from controller.navigation import cm_to_pixels

    def fake_px(cm_coord):
        if not cm_coord: return None
        px = cm_to_pixels((cm_coord[0], cm_coord[1]), WARPED_WIDTH, WARPED_HEIGHT, FIELD_WIDTH_CM, FIELD_HEIGHT_CM)
        if len(cm_coord) > 2:
            return (px[0], px[1], cm_coord[2])
        return (px[0], px[1])

    # gen fake yolo px coords to match nearest.py expecations
    world.robot_px = fake_px(world.robot)
    world.cross_px = fake_px(world.cross)
    world.ob_px = fake_px(world.ob)
    world.white_balls_px = [fake_px(b) for b in world.white_balls]
    world.white_wall_balls_px = [fake_px(b) for b in world.white_wall_balls]
    world.white_corner_balls_px = [fake_px(b) for b in world.white_corner_balls]

    # --- start sim ---
    plt.ion() # enable interactiv mode so plot updates without closing window
    fig, ax = plt.subplots(figsize=(10, 7))
    
    step_count = 1
    while True:
        # draw scene and calc
        target = draw_state(ax, world)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        if not target:
            print("All balls colected! Sim done.")
            ax.set_title("Sim done - All balls collected!")
            fig.canvas.draw()
            break
            
        print("Ready for next move! click window (or press key) to move fwd...")
        ax.set_title("click window (or press key) to take next ball!")
        fig.canvas.draw()
        
        # wait for user press before code continous
        plt.waitforbuttonpress()
            
        print(f"--- moving to ball #{step_count}: {target['id']} ---")
        
        # simulate robot driving to ball
        pos_cm = target.get("pos", target.get("pos_cm"))
        pos_px = target.get("pos_px")
        
        # update robot angle before moving its pos
        if world.robot and pos_cm:
            dx = pos_cm[0] - world.robot[0]
            dy = pos_cm[1] - world.robot[1]
            world.robot_angle = math.degrees(math.atan2(dy, dx))
            
        world.robot = pos_cm # move bot
        world.robot_px = pos_px
        
        # pick up ball (remove from arrays)
        if target["id"] == "goal":
            print("Target reached! Sim done.")
            ax.set_title("Sim done - target reached!")
            fig.canvas.draw()
            break
        elif target.get("type") == "staging":
            print(f"  -> staging point reached: {target['id']}")
            # we dont delete ball since we are just at a staging point
        elif target["id"] == "ob":
            world.ob = None
            world.ob_px = None
        else:
            # delete ball matching pos
            world.white_balls = [b for b in world.white_balls if (b[0], b[1]) != pos_cm]
            world.white_balls_px = [b for b in world.white_balls_px if (b[0], b[1]) != pos_px]
            
            world.white_wall_balls = [b for b in world.white_wall_balls if (b[0], b[1]) != pos_cm]
            world.white_wall_balls_px = [b for b in world.white_wall_balls_px if (b[0], b[1]) != pos_px]
            
            world.white_corner_balls = [b for b in world.white_corner_balls if (b[0], b[1]) != pos_cm]
            world.white_corner_balls_px = [b for b in world.white_corner_balls_px if (b[0], b[1]) != pos_px]
            
        step_count += 1

    plt.ioff() # disable interactive mode
    plt.show() # keep last frame open

if __name__ == "__main__":
    main()
