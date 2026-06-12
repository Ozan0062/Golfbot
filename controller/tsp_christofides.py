"""
tsp_christofides.py  —  Christofides 1.5-approximation for TSP.

Given a list of 2-D points, returns an ordered visit sequence (list of
indices) that starts from index 0 (the robot's current position).

Algorithm steps:
  1. Build complete weighted graph on all points (Euclidean distances).
  2. Compute minimum spanning tree T.
  3. Find odd-degree vertices O in T.
  4. Find minimum-weight perfect matching M on the subgraph induced by O.
  5. Combine T ∪ M into multigraph H (all vertices now have even degree).
  6. Find Eulerian circuit on H starting from node 0.
  7. Shortcut repeated nodes → Hamiltonian path.

The return leg (back to start) is dropped — the robot doesn't need to
return to its starting position after collecting all balls.

Requires: networkx  (pip install networkx --break-system-packages)
"""

import math
from itertools import combinations

import networkx as nx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def christofides_route(points: list) -> list:
    """
    Return visit order as a list of indices into *points*.

    points[0]  — robot's current position (always the tour start)
    points[1:] — targets (ball positions)

    Returns:
        [0, i, j, ...]  — visit order, length == len(points)
        []              — if points is empty
    """
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]

    # ── 1. Complete weighted graph ───────────────────────────────────────────
    G = nx.Graph()
    for i, j in combinations(range(n), 2):
        G.add_edge(i, j, weight=_dist(points[i], points[j]))

    # ── 2. Minimum spanning tree ─────────────────────────────────────────────
    T = nx.minimum_spanning_tree(G, weight="weight")

    # ── 3. Odd-degree vertices ───────────────────────────────────────────────
    odd = [v for v, deg in T.degree() if deg % 2 == 1]

    # ── 4. Min-weight perfect matching on odd-degree subgraph ────────────────
    odd_G = nx.Graph()
    for u, v in combinations(odd, 2):
        odd_G.add_edge(u, v, weight=_dist(points[u], points[v]))

    try:
        # networkx >= 2.6
        matching = nx.min_weight_matching(odd_G)
    except AttributeError:
        # Older networkx: negate weights and use max_weight_matching
        for u, v in odd_G.edges():
            odd_G[u][v]["weight"] = -odd_G[u][v]["weight"]
        matching = nx.max_weight_matching(odd_G, maxcardinality=True)

    # ── 5. Multigraph H = T ∪ matching ───────────────────────────────────────
    H = nx.MultiGraph(T)
    for u, v in matching:
        H.add_edge(u, v)

    # ── 6. Eulerian circuit from node 0 ──────────────────────────────────────
    euler = list(nx.eulerian_circuit(H, source=0))

    # ── 7. Shortcut → Hamiltonian path ───────────────────────────────────────
    seen = set()
    path = []
    for u, _ in euler:
        if u not in seen:
            path.append(u)
            seen.add(u)
    # Catch the very last node if the circuit ends on an unvisited one
    if euler and euler[-1][1] not in seen:
        path.append(euler[-1][1])

    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
