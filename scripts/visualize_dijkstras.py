import sys
import os
import matplotlib.pyplot as plt
import networkx as nx

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.tracker import WorldState
from controller.dijkstras import create_nodes_and_edges, calculate_best_route

def draw_state(ax, world):
    """Tegner det nuværende state og returnerer den beregnede rute."""
    ax.clear()
    
    # 1. Byg Graf og Beregn Rute
    G = create_nodes_and_edges(world)
    best_route = calculate_best_route(G)

    # 2. Opsætning af plottet
    ax.set_title("Dijkstra / NetworkX - Dynamisk Simulering (Luk vinduet for at stoppe)")
    ax.set_xlim(-10, 190) # Field width + margin
    ax.set_ylim(130, -10) # Field height + margin (inverted Y)
    
    # Tegn selve banens kanter (væggene)
    ax.plot([0, 180, 180, 0, 0], [0, 0, 120, 120, 0], color="black", linewidth=5, zorder=1, label="Væg/Bane-kant")
    
    # 3. Tegn noder
    pos = nx.get_node_attributes(G, 'pos')
    
    color_map = {
        "robot": "black",
        "open": "lightgray",
        "wall": "blue",
        "corner": "red",
        "obstacle": "purple"
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
            ax.text(data["pos"][0], data["pos"][1] + 6, f"+{data['penalty']} straf", fontsize=8, color="red", ha="center")

    # Tegn krydset manuelt (da det ikke længere er en node)
    if world.cross:
        ax.scatter(world.cross[0], world.cross[1], c="purple", s=200, zorder=5, label="obstacle", edgecolors="black")
        ax.text(world.cross[0], world.cross[1] - 4, "cross", fontsize=9, ha="center")

    # 4. Tegn pilene i fortløbende rækkefølge
    prev_pos = pos.get("robot", None)
    if prev_pos:
        num_steps = len(best_route)
        for i, target in enumerate(best_route):
            target_pos = target["pos"]
            
            # Gråskala gradient pil
            shade_value = 0.3 + 0.7 * (i / max(1, num_steps - 1))
            arrow_color = plt.cm.Greys(shade_value)
            
            ax.annotate("",
                        xy=target_pos, xycoords='data',
                        xytext=prev_pos, textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=arrow_color, alpha=0.9, lw=3, shrinkA=12, shrinkB=12))
                                        
            # Label med step nummer over noden
            ax.text(target_pos[0], target_pos[1] - 9, f"Step {i+1}", 
                    color="black", fontsize=9, fontweight="bold", ha="center", va="center",
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
                    
            prev_pos = target_pos

    # 5. Afsluttende grafik-indstillinger
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    return best_route

def main():
    # Definer vores test-bane
    world = WorldState(
        robot=(20.0, 60.0), # Robot starter i venstre side
        white_balls=[
            (90.0, 30.0, "open"),   # Over krydset
            (90.0, 90.0, "open"),   # Under krydset
            (60.0, 60.0, "open"),   # Venstre for krydset
            (130.0, 60.0, "open")   # Højre for krydset
        ],
        white_wall_balls=[
            (10.0, 60.0, "wall"),   # Venstre væg
            (170.0, 60.0, "wall")   # Højre væg
        ],
        white_corner_balls=[
            (10.0, 10.0, "corner"),   # Øverste venstre hjørne
            (170.0, 110.0, "corner")  # Nederste højre hjørne
        ],
        ob=(40.0, 40.0, "open"),
        cross=(90.0, 60.0) # Krydset placeret midt på banen!
    )

    # --- Start Simulationen ---
    plt.ion() # Slå interaktiv mode til, så plottet opdaterer uden at vi lukker vinduet
    fig, ax = plt.subplots(figsize=(10, 7))
    
    step_count = 1
    while True:
        # Tegn scenen og udregn
        best_route = draw_state(ax, world)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        if not best_route:
            print("Alle bolde er samlet op! Simulering færdig.")
            ax.set_title("Simulering færdig - Alle bolde opsamlet!")
            fig.canvas.draw()
            break
            
        print("Klar til næste træk! Klik i vinduet (eller tryk på en tast) for at køre frem...")
        ax.set_title("Klik i vinduet (eller tryk på en tast) for at tage næste bold!")
        fig.canvas.draw()
        
        # Vent på at brugeren trykker før koden fortsætter
        plt.waitforbuttonpress()
            
        print(f"--- Kører til bold #{step_count}: {best_route[0]['id']} ---")
        
        # Simuler at robotten kører hen til bolden
        target = best_route[0]
        world.robot = target["pos"] # Flyt robot
        pos_cm = target["pos"]
        
        # Saml bolden op (fjern den fra arrays)
        if target["id"] == "ob":
            world.ob = None
        else:
            # Slet bolden der matcher positionen
            world.white_balls = [b for b in world.white_balls if (b[0], b[1]) != pos_cm]
            world.white_wall_balls = [b for b in world.white_wall_balls if (b[0], b[1]) != pos_cm]
            world.white_corner_balls = [b for b in world.white_corner_balls if (b[0], b[1]) != pos_cm]
            
        step_count += 1

    plt.ioff() # Sluk interaktiv mode
    plt.show() # Hold det sidste frame åbent

if __name__ == "__main__":
    main()
