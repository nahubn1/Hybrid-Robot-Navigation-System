from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from data_generation.utility_ground_truth import (
    Rectangle, Environment,
    heuristic_sampling, build_prm, exhaustive_prm_runs, connectivity_high_utility_regions
)


def create_simple_env():
    obs = [Rectangle(-0.5, -0.1, 0.5, 0.1)]
    return Environment(area_size=4.0, obstacles=obs)


def test_heuristic_sampling():
    env = create_simple_env()
    pts = heuristic_sampling(env, grid_step=1.0)
    assert len(pts) > 0
    for x, y in pts:
        assert env.collision_free((x, y))


def test_exhaustive_prm_runs():
    env = create_simple_env()
    samples, graph = exhaustive_prm_runs(env, num_samples=30, num_runs=5, radius=1.5)
    assert isinstance(samples, list)
    assert isinstance(graph.number_of_nodes(), int)
    if samples:
        assert all(len(pt) == 2 for pt in samples)


def test_connectivity_high_utility_regions():
    env = create_simple_env()
    graph = build_prm(env, num_samples=20, radius=1.5)
    top_pts = connectivity_high_utility_regions(graph, top_k=5)
    assert len(top_pts) <= 5
    for pt in top_pts:
        assert len(pt) == 2
