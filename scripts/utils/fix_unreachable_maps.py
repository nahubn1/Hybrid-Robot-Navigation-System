import argparse
from pathlib import Path
import numpy as np
from collections import deque
import random


def path_exists(grid: np.ndarray) -> bool:
    start_pos = np.argwhere(grid == 8)
    goal_pos = np.argwhere(grid == 9)
    if start_pos.size == 0 or goal_pos.size == 0:
        return False
    start = tuple(start_pos[0])
    goal = tuple(goal_pos[0])
    q = deque([start])
    visited = {start}
    free = lambda y, x: grid[y, x] in (0, 8, 9)
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            return True
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                if free(ny, nx) and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    q.append((ny, nx))
    return False


def random_free_cell(grid: np.ndarray) -> tuple[int, int]:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise ValueError("No free cells available")
    idx = random.randrange(len(free))
    y, x = free[idx]
    return int(y), int(x)


def fix_file(path: Path) -> bool:
    data = np.load(path, allow_pickle=True)
    grid = data["map"].copy()
    start_pos = tuple(np.argwhere(grid == 8)[0])
    goal_pos = tuple(np.argwhere(grid == 9)[0])
    if path_exists(grid):
        return False
    # remove old markers
    grid[start_pos] = 0
    grid[goal_pos] = 0
    for _ in range(1000):
        new_start = random_free_cell(grid)
        new_goal = random_free_cell(grid)
        if new_start == new_goal:
            continue
        grid[new_start] = 8
        grid[new_goal] = 9
        if path_exists(grid):
            data_dict = dict(data)
            data_dict["map"] = grid
            np.savez(path, **data_dict)
            print(f"Edited {path}: start {start_pos}->{new_start}, goal {goal_pos}->{new_goal}")
            return True
        grid[new_start] = 0
        grid[new_goal] = 0
    print(f"Could not find new positions for {path}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix maps without valid paths")
    parser.add_argument("directory", type=Path, help="Directory with .npz maps")
    args = parser.parse_args()
    for p in sorted(args.directory.glob("*.npz")):
        fix_file(p)


if __name__ == "__main__":
    main()
