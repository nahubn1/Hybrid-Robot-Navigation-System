from pathlib import Path
import sys

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from data_generation.ground_truth_heatmaps import (
    PRMConfig,
    HeatmapConfig,
    build_prm,
    betweenness_high_utility_nodes,
    largest_component_nodes,
    nodes_to_heatmap,
)


def test_prm_and_betweenness():
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[5:15, 10] = 1
    prm = build_prm(grid, PRMConfig(num_samples=40, radius=5))
    points = betweenness_high_utility_nodes(prm, top_k=5)
    assert len(points) <= 5
    for y, x in points:
        assert 0 <= x < grid.shape[1]
        assert 0 <= y < grid.shape[0]


def test_largest_component_nodes():
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[10:, :] = 1
    prm = build_prm(grid, PRMConfig(num_samples=30, radius=4))
    pts = largest_component_nodes(prm)
    assert all(len(p) == 2 for p in pts)


def test_nodes_to_heatmap():
    shape = (10, 10)
    points = [(5, 5), (2, 2)]
    heatmap = nodes_to_heatmap(points, shape, sigma=1.0)
    assert heatmap.shape == shape
    assert heatmap.max() <= 1.0
    assert heatmap.min() >= 0.0
