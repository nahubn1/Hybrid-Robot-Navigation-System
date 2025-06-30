import random
from typing import List, Dict, Tuple

import networkx as nx

from .office_map_generator import Leaf, Wall


def build_room_adjacency_graph(leaves: List[Leaf], min_length: int = 3) -> nx.Graph:
    """Construct a graph where nodes are rooms and edges represent shared walls."""
    g = nx.Graph()
    for leaf in leaves:
        g.add_node(leaf.id, room=leaf)

    for i in range(len(leaves)):
        r1 = leaves[i]
        for j in range(i + 1, len(leaves)):
            r2 = leaves[j]
            # vertical adjacency
            if r1.x + r1.w == r2.x or r2.x + r2.w == r1.x:
                x = r1.x + r1.w if r1.x + r1.w == r2.x else r2.x + r2.w
                start = max(r1.y, r2.y)
                end = min(r1.y + r1.h, r2.y + r2.h)
                if end - start >= min_length:
                    g.add_edge(r1.id, r2.id, wall=Wall('v', x, start, end))
            # horizontal adjacency
            if r1.y + r1.h == r2.y or r2.y + r2.h == r1.y:
                y = r1.y + r1.h if r1.y + r1.h == r2.y else r2.y + r2.h
                start = max(r1.x, r2.x)
                end = min(r1.x + r1.w, r2.x + r2.w)
                if end - start >= min_length:
                    g.add_edge(r1.id, r2.id, wall=Wall('h', y, start, end))
    return g


def _draw_single_door(grid, wall: Wall, walls: List[Wall], wall_thickness: int, door_cfg: Dict, rng: random.Random) -> None:
    half = wall_thickness // 2
    door_min, door_max = door_cfg.get("width_range", [3, 5])
    door_width = rng.randint(door_min, door_max)

    if wall.orientation == 'v':
        x = wall.pos
        y0, y1 = wall.start, wall.end
        low = y0 + half + 1
        high = max(low, y1 - half - door_width)
        if high <= low:
            return
        for _ in range(10):
            door_y = rng.randint(low, high)
            conflict = False
            for other in walls:
                if other.orientation == 'h' and other.start <= x < other.end:
                    if other.pos >= door_y - half and other.pos < door_y + door_width + half:
                        conflict = True
                        break
            if not conflict:
                grid[door_y:door_y + door_width, max(0, x - half):min(grid.shape[1], x + half + 1)] = 0
                break
    else:
        y = wall.pos
        x0, x1 = wall.start, wall.end
        low = x0 + half + 1
        high = max(low, x1 - half - door_width)
        if high <= low:
            return
        for _ in range(10):
            door_x = rng.randint(low, high)
            conflict = False
            for other in walls:
                if other.orientation == 'v' and other.start <= y < other.end:
                    if other.pos >= door_x - half and other.pos < door_x + door_width + half:
                        conflict = True
                        break
            if not conflict:
                grid[max(0, y - half):min(grid.shape[0], y + half + 1), door_x:door_x + door_width] = 0
                break


def place_doors_from_graph(grid, graph: nx.Graph, walls: List[Wall], wall_thickness: int, door_cfg: Dict, rng: random.Random, additional_prob: float = 0.0) -> None:
    """Place doors ensuring connectivity then optionally add extra doors."""
    if not graph.nodes:
        return

    visited = set()
    queue = [next(iter(graph.nodes))]
    visited.add(queue[0])
    used_edges = set()

    while queue:
        n = queue.pop(0)
        for nbr in graph.neighbors(n):
            if nbr not in visited:
                wall = graph.edges[n, nbr]['wall']
                _draw_single_door(grid, wall, walls, wall_thickness, door_cfg, rng)
                used_edges.add((n, nbr))
                used_edges.add((nbr, n))
                visited.add(nbr)
                queue.append(nbr)

    for u, v, data in graph.edges(data=True):
        if (u, v) in used_edges:
            continue
        if rng.random() < additional_prob:
            wall = data['wall']
            _draw_single_door(grid, wall, walls, wall_thickness, door_cfg, rng)

