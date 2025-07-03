from pathlib import Path
import numpy as np
import sys

SRC_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_PATH))

from scripts.utils.fix_unreachable_maps import path_exists, fix_file


def test_path_exists_blocked():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[0, 0] = 8
    grid[4, 4] = 9
    assert path_exists(grid)
    grid[:, 2] = 1
    assert not path_exists(grid)


def test_fix_file(tmp_path):
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[0, 0] = 8
    grid[4, 4] = 9
    grid[:, 2] = 1
    path = tmp_path / "sample.npz"
    np.savez(path, map=grid, clearance=0.1, step_size=1.0, config="{}")
    assert not path_exists(grid)
    assert fix_file(path)
    data = np.load(path)
    new_grid = data["map"]
    assert path_exists(new_grid)
    start = tuple(np.argwhere(new_grid == 8)[0])
    goal = tuple(np.argwhere(new_grid == 9)[0])
    assert start != goal
