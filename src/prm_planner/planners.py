import math
from typing import List, Tuple

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy import ndimage


class BasePRMPlanner:
    """Shared functionality for PRM based planners."""

    def __init__(self, num_samples: int = 200, connection_radius: float = 10.0) -> None:
        self.num_samples = int(num_samples)
        self.connection_radius = float(connection_radius)
        self.sampled_nodes: List[Tuple[int, int]] = []
        self.roadmap: nx.Graph | None = None
        self.path: List[Tuple[int, int]] = []

    # --- Utility methods -------------------------------------------------
    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int):
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

    @classmethod
    def _is_collision_free(cls, p: Tuple[int, int], q: Tuple[int, int], grid: np.ndarray) -> bool:
        for x, y in cls._bresenham_line(p[0], p[1], q[0], q[1]):
            if grid[y, x] not in (0, 8, 9):
                return False
        return True

    # --- Core roadmap methods -------------------------------------------
    def _build_roadmap(self, nodes: List[Tuple[int, int]], grid: np.ndarray) -> nx.Graph:
        tree = KDTree(nodes)
        G = nx.Graph()
        for idx, pos in enumerate(nodes):
            G.add_node(idx, pos=pos)
        for i, p in enumerate(nodes):
            neighbors = tree.query_ball_point(p, r=self.connection_radius)
            for j in neighbors:
                if j <= i:
                    continue
                q = nodes[j]
                if self._is_collision_free(p, q, grid):
                    dist = float(math.hypot(p[0] - q[0], p[1] - q[1]))
                    G.add_edge(i, j, weight=dist)
        return G

    def _find_path(self, graph: nx.Graph, start_id: int, goal_id: int) -> List[Tuple[int, int]]:
        def heuristic(u: int, v: int) -> float:
            pu = graph.nodes[u]['pos']
            pv = graph.nodes[v]['pos']
            return math.hypot(pu[0] - pv[0], pu[1] - pv[1])

        try:
            idx_path = nx.astar_path(graph, start_id, goal_id, heuristic=heuristic, weight='weight')
        except nx.NetworkXNoPath:
            return []
        return [graph.nodes[i]['pos'] for i in idx_path]

    # --- Abstract methods ------------------------------------------------
    def _sample_nodes(self, grid: np.ndarray, clearance: float, *args) -> List[Tuple[int, int]]:
        raise NotImplementedError

    # --- Public interface -------------------------------------------------
    def plan(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int], grid: np.ndarray,
             clearance: float, *args) -> List[Tuple[int, int]]:
        self.sampled_nodes = self._sample_nodes(grid, clearance, *args)
        nodes = list(self.sampled_nodes)
        nodes.extend([start_pos, goal_pos])
        start_id = len(nodes) - 2
        goal_id = len(nodes) - 1
        self.roadmap = self._build_roadmap(nodes, grid)
        self.path = self._find_path(self.roadmap, start_id, goal_id)
        return self.path


class PRMPlanner(BasePRMPlanner):
    """Classical PRM planner using uniform random sampling."""

    def _sample_nodes(self, grid: np.ndarray, clearance: float, *args) -> List[Tuple[int, int]]:
        traversable = np.isin(grid, (0, 8, 9))
        dist = ndimage.distance_transform_edt(traversable)
        valid = dist >= clearance
        h, w = grid.shape
        nodes: List[Tuple[int, int]] = []
        while len(nodes) < self.num_samples:
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            if valid[y, x]:
                nodes.append((x, y))
        return nodes


class DNNPRMPlanner(PRMPlanner):
    """PRM planner guided by a trained neural network."""

    def __init__(self, num_samples: int, connection_radius: float, handler, epsilon: float = 0.05) -> None:
        super().__init__(num_samples, connection_radius)
        self.handler = handler
        self.epsilon = float(epsilon)

    def _sample_nodes(self, grid: np.ndarray, clearance: float, step_size: float | None = None) -> List[Tuple[int, int]]:
        step = 0.0 if step_size is None else float(step_size)
        robot = np.array([float(clearance), step], dtype=np.float32)
        heatmap = self.handler.predict(grid, robot)

        traversable = np.isin(grid, (0, 8, 9))
        dist = ndimage.distance_transform_edt(traversable)
        valid = dist >= clearance

        probs = heatmap.astype(float)
        probs[~valid] = 0.0
        flat = probs.ravel()
        if flat.sum() > 0:
            flat /= flat.sum()
        else:
            flat = None

        h, w = grid.shape
        nodes: List[Tuple[int, int]] = []
        while len(nodes) < self.num_samples:
            if flat is None or np.random.random() < self.epsilon:
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
            else:
                idx = np.random.choice(h * w, p=flat)
                y, x = divmod(idx, w)
            if valid[y, x]:
                nodes.append((x, y))
        return nodes
