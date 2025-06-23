from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import math
import networkx as nx


@dataclass
class Rectangle:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class Environment:
    area_size: float
    obstacles: List[Rectangle]

    def in_bounds(self, point: Tuple[float, float]) -> bool:
        half = self.area_size / 2.0
        x, y = point
        return -half <= x <= half and -half <= y <= half

    def collision_free(self, point: Tuple[float, float]) -> bool:
        if not self.in_bounds(point):
            return False
        return not any(obs.contains(point) for obs in self.obstacles)


def line_intersects_rect(p1: Tuple[float, float], p2: Tuple[float, float], rect: Rectangle) -> bool:
    """Check if a line segment intersects an axis-aligned rectangle."""
    x1, y1 = p1
    x2, y2 = p2
    p = [-(x2 - x1), x2 - x1, -(y2 - y1), y2 - y1]
    q = [x1 - rect.xmin, rect.xmax - x1, y1 - rect.ymin, rect.ymax - y1]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return True if u1 < u2 else False


def line_collision_free(env: Environment, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
    if not env.in_bounds(p1) or not env.in_bounds(p2):
        return False
    for obs in env.obstacles:
        if line_intersects_rect(p1, p2, obs):
            return False
    return True


def heuristic_sampling(env: Environment, grid_step: float = 0.5, narrow_thresh: float = 1.0) -> List[Tuple[float, float]]:
    """Generate sample points using simple heuristics."""
    samples = []
    half = env.area_size / 2.0
    coords = [(-half + i * grid_step) for i in range(int(env.area_size / grid_step) + 1)]
    for x in coords:
        for y in coords:
            pt = (x, y)
            if env.collision_free(pt):
                samples.append(pt)
    # Add midpoints of close obstacle pairs
    for i, obs_a in enumerate(env.obstacles):
        for obs_b in env.obstacles[i + 1:]:
            dx = max(0, max(obs_a.xmin, obs_b.xmin) - min(obs_a.xmax, obs_b.xmax))
            dy = max(0, max(obs_a.ymin, obs_b.ymin) - min(obs_a.ymax, obs_b.ymax))
            gap = math.hypot(dx, dy)
            if 0 < gap < narrow_thresh:
                mx = (obs_a.xmax + obs_b.xmin) / 2.0
                my = (obs_a.ymax + obs_b.ymin) / 2.0
                midpoint = (mx, my)
                if env.collision_free(midpoint):
                    samples.append(midpoint)
    return samples


def build_prm(env: Environment, num_samples: int = 100, radius: float = 1.5) -> nx.Graph:
    nodes: List[Tuple[float, float]] = []
    while len(nodes) < num_samples:
        x = random.uniform(-env.area_size / 2.0, env.area_size / 2.0)
        y = random.uniform(-env.area_size / 2.0, env.area_size / 2.0)
        pt = (x, y)
        if env.collision_free(pt):
            nodes.append(pt)
    G = nx.Graph()
    for idx, p in enumerate(nodes):
        G.add_node(idx, pos=p)
    for i, p in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            q = nodes[j]
            if math.dist(p, q) <= radius and line_collision_free(env, p, q):
                G.add_edge(i, j, weight=math.dist(p, q))
    return G


def exhaustive_prm_runs(env: Environment, num_samples: int = 100, num_runs: int = 20, radius: float = 1.5) -> Tuple[List[Tuple[float, float]], nx.Graph]:
    G = build_prm(env, num_samples, radius)
    usage = {n: 0 for n in G.nodes}
    for _ in range(num_runs):
        start = (random.uniform(-env.area_size/2, env.area_size/2), random.uniform(-env.area_size/2, env.area_size/2))
        goal = (random.uniform(-env.area_size/2, env.area_size/2), random.uniform(-env.area_size/2, env.area_size/2))
        if not env.collision_free(start) or not env.collision_free(goal):
            continue
        start_id = len(G.nodes)
        goal_id = start_id + 1
        G.add_node(start_id, pos=start)
        G.add_node(goal_id, pos=goal)
        for n, data in list(G.nodes(data=True)):
            if n in (start_id, goal_id):
                continue
            pos = data['pos']
            if math.dist(pos, start) <= radius and line_collision_free(env, pos, start):
                G.add_edge(start_id, n, weight=math.dist(pos, start))
            if math.dist(pos, goal) <= radius and line_collision_free(env, pos, goal):
                G.add_edge(goal_id, n, weight=math.dist(pos, goal))
        try:
            path = nx.shortest_path(G, start_id, goal_id, weight='weight')
        except nx.NetworkXNoPath:
            path = []
        for node in path:
            if node not in (start_id, goal_id):
                usage[node] += 1
        G.remove_node(start_id)
        G.remove_node(goal_id)
    sorted_nodes = sorted(usage.items(), key=lambda x: x[1], reverse=True)
    high_util = [G.nodes[n]['pos'] for n, count in sorted_nodes if count > 0]
    return high_util, G


def connectivity_high_utility_regions(G: nx.Graph, top_k: int = 10) -> List[Tuple[float, float]]:
    if G.number_of_nodes() == 0:
        return []
    centrality = nx.betweenness_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [G.nodes[n]['pos'] for n, _ in sorted_nodes[:top_k]]
