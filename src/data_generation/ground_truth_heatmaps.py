"""Utility functions for generating heatmaps from occupancy grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math

import numpy as np
import networkx as nx


@dataclass
class PRMConfig:
    """Configuration for probabilistic roadmap generation."""

    num_samples: int = 500
    radius: float = 10.0


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""

    sigma: float = 2.0
    top_k: int = 50


def _sample_free_points(grid: np.ndarray, num_samples: int) -> List[Tuple[int, int]]:
    free = np.argwhere(np.isin(grid, (0, 8, 9)))
    if len(free) == 0:
        return []
    idx = np.random.choice(len(free), size=min(num_samples, len(free)), replace=False)
    pts = [(int(y), int(x)) for y, x in free[idx]]
    return pts


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    """Yield coordinates of cells on a line using Bresenham algorithm."""
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


def _collision_free(grid: np.ndarray, p: Tuple[int, int], q: Tuple[int, int]) -> bool:
    for x, y in _bresenham_line(p[0], p[1], q[0], q[1]):
        if grid[y, x] not in (0, 8, 9):
            return False
    return True


def build_prm(grid: np.ndarray, config: PRMConfig) -> nx.Graph:
    """Build a simple PRM directly on the occupancy grid."""
    points = _sample_free_points(grid, config.num_samples)
    G = nx.Graph()
    for idx, (y, x) in enumerate(points):
        G.add_node(idx, pos=(x, y))
    for i, (y1, x1) in enumerate(points):
        for j in range(i + 1, len(points)):
            y2, x2 = points[j]
            if math.hypot(x1 - x2, y1 - y2) <= config.radius and _collision_free(
                grid, (x1, y1), (x2, y2)
            ):
                G.add_edge(i, j, weight=math.hypot(x1 - x2, y1 - y2))
    return G


def betweenness_high_utility_nodes(G: nx.Graph, top_k: int) -> List[Tuple[int, int]]:
    if G.number_of_nodes() == 0:
        return []
    centrality = nx.betweenness_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [G.nodes[n]["pos"][::-1] for n, _ in sorted_nodes[:top_k]]


def largest_component_nodes(G: nx.Graph) -> List[Tuple[int, int]]:
    if G.number_of_nodes() == 0:
        return []
    components = list(nx.connected_components(G))
    if not components:
        return []
    largest = max(components, key=len)
    return [G.nodes[n]["pos"][::-1] for n in largest]


def _gaussian_kernel(sigma: float) -> np.ndarray:
    size = max(1, int(6 * sigma + 1))
    if size % 2 == 0:
        size += 1
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel1d = np.exp(-(ax**2) / (2 * sigma**2))
    kernel1d = kernel1d / kernel1d.sum()
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / kernel2d.sum()


def nodes_to_heatmap(points: Iterable[Tuple[int, int]], shape: Tuple[int, int], sigma: float) -> np.ndarray:
    heatmap = np.zeros(shape, dtype=float)
    kernel = _gaussian_kernel(sigma)
    k_half = kernel.shape[0] // 2
    h, w = shape
    for y, x in points:
        x0 = int(round(x))
        y0 = int(round(y))
        x_start = max(0, x0 - k_half)
        x_end = min(w, x0 + k_half + 1)
        y_start = max(0, y0 - k_half)
        y_end = min(h, y0 + k_half + 1)
        kx_start = x_start - (x0 - k_half)
        kx_end = kx_start + (x_end - x_start)
        ky_start = y_start - (y0 - k_half)
        ky_end = ky_start + (y_end - y_start)
        heatmap[y_start:y_end, x_start:x_end] += kernel[ky_start:ky_end, kx_start:kx_end]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def generate_heatmap(grid: np.ndarray, prm_cfg: PRMConfig, hm_cfg: HeatmapConfig, method: str = "betweenness") -> np.ndarray:
    graph = build_prm(grid, prm_cfg)
    if method == "betweenness":
        points = betweenness_high_utility_nodes(graph, hm_cfg.top_k)
    else:
        points = largest_component_nodes(graph)
    return nodes_to_heatmap(points, grid.shape, hm_cfg.sigma)

