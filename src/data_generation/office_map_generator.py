"""Procedural office-like map generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np


@dataclass
class Leaf:
    """Leaf node of the BSP tree."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class Wall:
    orientation: str  # 'v' or 'h'
    pos: int
    start: int
    end: int


def bsp_partition(width: int, height: int, cfg: Dict, rng: random.Random) -> Tuple[List[Leaf], List[Wall]]:
    """Generate a BSP layout returning rooms and dividing walls."""
    min_leaf = int(cfg.get("min_leaf_size", 20))
    split_low, split_high = cfg.get("split_range", [0.4, 0.6])

    leaves: List[Leaf] = [Leaf(0, 0, width, height)]
    walls: List[Wall] = []
    queue = [0]
    while queue:
        idx = queue.pop()
        leaf = leaves[idx]
        if leaf.w < 2 * min_leaf and leaf.h < 2 * min_leaf:
            continue
        split_vert = leaf.w > leaf.h
        if leaf.w >= 2 * min_leaf and leaf.h >= 2 * min_leaf:
            split_vert = rng.random() < 0.5
        if split_vert:
            min_split = int(leaf.w * split_low)
            max_split = int(leaf.w * split_high)
            if max_split - min_split < min_leaf:
                continue
            split = rng.randint(min_split, max_split)
            left = Leaf(leaf.x, leaf.y, split, leaf.h)
            right = Leaf(leaf.x + split, leaf.y, leaf.w - split, leaf.h)
            leaves[idx] = left
            leaves.append(right)
            queue.append(len(leaves) - 1)
            queue.append(idx)
            walls.append(Wall('v', leaf.x + split, leaf.y, leaf.y + leaf.h))
        else:
            min_split = int(leaf.h * split_low)
            max_split = int(leaf.h * split_high)
            if max_split - min_split < min_leaf:
                continue
            split = rng.randint(min_split, max_split)
            top = Leaf(leaf.x, leaf.y, leaf.w, split)
            bottom = Leaf(leaf.x, leaf.y + split, leaf.w, leaf.h - split)
            leaves[idx] = top
            leaves.append(bottom)
            queue.append(len(leaves) - 1)
            queue.append(idx)
            walls.append(Wall('h', leaf.y + split, leaf.x, leaf.x + leaf.w))
    return leaves, walls


def apply_walls(grid: np.ndarray, walls: List[Wall], wall_thickness: int, door_cfg: Dict, rng: random.Random) -> None:
    """Draw walls and doorways onto the grid."""
    half = wall_thickness // 2
    for wall in walls:
        door_min, door_max = door_cfg.get("width_range", [3, 5])
        door_width = rng.randint(door_min, door_max)
        if wall.orientation == 'v':
            y0, y1 = wall.start, wall.end
            x = wall.pos
            grid[y0:y1, max(0, x - half):min(grid.shape[1], x + half + 1)] = 1
            door_y = rng.randint(y0 + half + 1, max(y0 + half + 1, y1 - half - door_width))
            grid[door_y:door_y + door_width, max(0, x - half):min(grid.shape[1], x + half + 1)] = 0
        else:
            x0, x1 = wall.start, wall.end
            y = wall.pos
            grid[max(0, y - half):min(grid.shape[0], y + half + 1), x0:x1] = 1
            door_x = rng.randint(x0 + half + 1, max(x0 + half + 1, x1 - half - door_width))
            grid[max(0, y - half):min(grid.shape[0], y + half + 1), door_x:door_x + door_width] = 0


def draw_square(grid, x, y, size):
    grid[y:y + size, x:x + size] = 1


def draw_pyramid(grid, x, y, size):
    # Draw a right triangle (pyramid base) with right angle at (x, y)
    for i in range(size):
        grid[y + i, x:x + i + 1] = 1


def draw_cylinder(grid, x, y, size):
    # Draw a filled circle (cylinder base) centered in the square
    cy = y + size // 2
    cx = x + size // 2
    r = size // 2
    for iy in range(y, y + size):
        for ix in range(x, x + size):
            if (ix - cx) ** 2 + (iy - cy) ** 2 <= r ** 2:
                grid[iy, ix] = 1


def draw_rectangle(grid, x, y, size):
    grid[y:y + size, x:x + size * 2] = 2


def draw_triangle(grid, x, y, size):
    for i in range(size):
        grid[y + i, x:x + i + 1] = 3


def draw_circle(grid, x, y, size):
    cy = y + size // 2
    cx = x + size // 2
    r = size // 2
    for iy in range(y, y + size):
        for ix in range(x, x + size):
            if (ix - cx) ** 2 + (iy - cy) ** 2 <= r ** 2:
                grid[iy, ix] = 4


def draw_u_shape(grid, x, y, size):
    # U-shape: three sides of a square
    grid[y:y + size, x] = 5
    grid[y:y + size, x + size - 1] = 5
    grid[y + size - 1, x:x + size] = 5


def draw_l_shape(grid, x, y, size):
    # L-shape: two sides of a square
    grid[y:y + size, x] = 6
    grid[y + size - 1, x:x + size] = 6


def draw_t_shape(grid, x, y, size):
    # T-shape: vertical and horizontal bar
    grid[y:y + size, x + size // 2] = 7
    grid[y + size // 2, x:x + size] = 7


def place_obstacles(grid: np.ndarray, rooms: List[Leaf], cfg: Dict, wall_thickness: int, rng: random.Random) -> None:
    count_min, count_max = cfg.get("count_range_per_room", [0, 0])
    door_clearance = int(cfg.get("door_clearance", 1))
    obstacle_clearance = int(cfg.get("obstacle_clearance", 2))
    shape_defs = cfg.get("shape_definitions", [
        {"type": "rectangle", "size_range": [2, 4], "weight": 1.0},
        {"type": "triangle", "size_range": [2, 4], "weight": 1.0},
        {"type": "u-shape", "size_range": [2, 4], "weight": 1.0},
        {"type": "l-shape", "size_range": [2, 4], "weight": 1.0},
        {"type": "t-shape", "size_range": [2, 4], "weight": 1.0},
    ])
    weights = [sd.get("weight", 1.0) for sd in shape_defs]
    for room in rooms:
        num = rng.randint(count_min, count_max + 1)
        placed = []
        for _ in range(num):
            shape_def = rng.choices(shape_defs, weights)[0]
            size_min, size_max = shape_def.get("size_range", [2, 4])
            size = rng.randint(size_min, size_max)
            shape_type = shape_def.get("type", "rectangle")
            x_min = room.x + wall_thickness
            x_max = room.x + room.w - size - wall_thickness
            y_min = room.y + wall_thickness
            y_max = room.y + room.h - size - wall_thickness
            if x_min > x_max or y_min > y_max:
                continue
            for _ in range(20):
                x = rng.randint(x_min, x_max)
                y = rng.randint(y_min, y_max)
                patch = grid[y - door_clearance:y + size + door_clearance, x - door_clearance:x + size + door_clearance]
                # Check for wall collision (walls are 1)
                if np.any(patch == 1):
                    continue
                # Check for obstacle collision (obstacles are 2-7)
                if np.any(patch >= 2):
                    continue
                # Check for clearance with other obstacles
                for ox, oy, osize in placed:
                    if abs(x - ox) < osize + size + obstacle_clearance and abs(y - oy) < osize + size + obstacle_clearance:
                        break
                else:
                    # Final check: ensure the obstacle's area does not overlap any wall
                    obstacle_area = grid[y:y + size, x:x + size]
                    if np.any(obstacle_area == 1):
                        continue
                    if shape_type == "rectangle":
                        draw_rectangle(grid, x, y, size)
                    elif shape_type == "triangle":
                        draw_triangle(grid, x, y, size)
                    elif shape_type == "u-shape":
                        draw_u_shape(grid, x, y, size)
                    elif shape_type == "l-shape":
                        draw_l_shape(grid, x, y, size)
                    elif shape_type == "t-shape":
                        draw_t_shape(grid, x, y, size)
                    else:
                        draw_rectangle(grid, x, y, size)
                    placed.append((x, y, size))
                    break


def flood_fill_connected(grid: np.ndarray) -> bool:
    """Check if free space is a single connected component."""
    free = np.argwhere(grid == 0)
    if len(free) == 0:
        return False
    start = tuple(free[0])
    visited = np.zeros_like(grid, dtype=bool)
    stack = [start]
    visited[start] = True
    while stack:
        cy, cx = stack.pop()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                if not visited[ny, nx] and grid[ny, nx] == 0:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return visited[grid == 0].all()


def generate_office_map(cfg: Dict, rng: random.Random) -> np.ndarray:
    resolution = int(cfg.get("map_resolution", 200))
    shell = int(cfg.get("shell_thickness", 2))
    wall_thickness = int(cfg.get("bsp", {}).get("wall_thickness", 2))

    grid = np.zeros((resolution, resolution), dtype=np.uint8)
    grid[:shell, :] = 1
    grid[-shell:, :] = 1
    grid[:, :shell] = 1
    grid[:, -shell:] = 1

    rooms, walls = bsp_partition(resolution - 2 * shell, resolution - 2 * shell, cfg.get("bsp", {}), rng)
    shifted_rooms = [Leaf(r.x + shell, r.y + shell, r.w, r.h) for r in rooms]
    shifted_walls = [Wall(w.orientation, w.pos + shell if w.orientation == 'v' else w.pos + shell, w.start + shell, w.end + shell) for w in walls]

    apply_walls(grid, shifted_walls, wall_thickness, cfg.get("doors", {}), rng)
    place_obstacles(grid, shifted_rooms, cfg.get("obstacles", {}), wall_thickness, rng)

    return grid

