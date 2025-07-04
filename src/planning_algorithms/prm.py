import math
from typing import Tuple, List
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def sample_free_points(grid: np.ndarray, num_samples: int) -> List[Tuple[int, int]]:
    free = np.argwhere(np.isin(grid, (0, 8, 9)))
    if len(free) == 0:
        return []
    if len(free) <= num_samples:
        idx = np.arange(len(free))
    else:
        idx = np.random.choice(len(free), size=num_samples, replace=False)
    pts = [(int(x), int(y)) for y, x in free[idx]]
    return pts


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def is_collision_free(grid: np.ndarray, p: Tuple[int, int], q: Tuple[int, int]) -> bool:
    for x, y in bresenham_line(p[0], p[1], q[0], q[1]):
        if grid[y, x] not in (0, 8, 9):
            return False
    return True


def build_prm(grid: np.ndarray, num_samples: int = 500, k: int = 10) -> nx.Graph:
    nodes = sample_free_points(grid, num_samples)
    if not nodes:
        return nx.Graph()
    tree = cKDTree(nodes)
    G = nx.Graph()
    for idx, (x, y) in enumerate(nodes):
        G.add_node(idx, pos=(x, y))
    for i, (x, y) in enumerate(nodes):
        dists, idxs = tree.query((x, y), k=k + 1)
        for dist, j in zip(dists[1:], idxs[1:]):
            if j >= len(nodes):
                continue
            x2, y2 = nodes[j]
            if is_collision_free(grid, (x, y), (x2, y2)):
                G.add_edge(i, j, weight=float(math.hypot(x - x2, y - y2)))
    return G
