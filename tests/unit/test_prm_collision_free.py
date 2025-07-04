from pathlib import Path
import sys

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from planning_algorithms.prm import is_collision_free, sample_free_points


def test_start_goal_cells_allowed():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 8
    assert is_collision_free(grid, (0, 0), (4, 4))
    pts = sample_free_points(grid, 30)
    assert (2, 2) in pts
    grid[2, 2] = 9
    assert is_collision_free(grid, (0, 0), (4, 4))
    pts = sample_free_points(grid, 30)
    assert (2, 2) in pts
