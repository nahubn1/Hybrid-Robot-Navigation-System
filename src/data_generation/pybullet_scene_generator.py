import random
import math
from pathlib import Path
from typing import List

import pybullet as p


def create_cluttered_scene(obstacle_count: int = 10,
                           area_size: float = 5.0,
                           size_range=(0.2, 0.5),
                           urdf_path: str = "",
                           seed: int = None) -> List[int]:
    """Spawn randomly placed cubes for a cluttered scene."""
    if seed is not None:
        random.seed(seed)
    ids = []
    half = area_size / 2.0
    for _ in range(obstacle_count):
        size = random.uniform(*size_range)
        x = random.uniform(-half, half)
        y = random.uniform(-half, half)
        z = size / 2.0
        if urdf_path:
            body = p.loadURDF(urdf_path, [x, y, z], globalScaling=size)
        else:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3,
                                      rgbaColor=[0.7, 0.7, 0.7, 1])
            body = p.createMultiBody(0, col, vis, [x, y, z])
        ids.append(body)
    return ids


def create_room_scene(num_rooms: int = 2,
                       area_size: float = 5.0,
                       wall_thickness: float = 0.1,
                       doorway: float = 0.8,
                       urdf_path: str = "",
                       seed: int = None) -> List[int]:
    """Create room-like structures using wall segments."""
    if seed is not None:
        random.seed(seed)
    ids = []
    grid = int(math.ceil(math.sqrt(num_rooms)))
    room_size = area_size / grid
    half = area_size / 2.0
    wall_len = room_size
    for i in range(grid):
        for j in range(grid):
            if len(ids) >= num_rooms * 4:
                break
            cx = -half + i * room_size
            cy = -half + j * room_size
            # four walls with a doorway in one random wall
            door_wall = random.choice([0, 1, 2, 3])
            for w in range(4):
                if w == door_wall:
                    continue
                if w == 0:  # bottom
                    pos = [cx + wall_len/2, cy, wall_thickness/2]
                    orn = p.getQuaternionFromEuler([0,0,0])
                    length = wall_len
                elif w == 1:  # top
                    pos = [cx + wall_len/2, cy + room_size, wall_thickness/2]
                    orn = p.getQuaternionFromEuler([0,0,0])
                    length = wall_len
                elif w == 2:  # left
                    pos = [cx, cy + wall_len/2, wall_thickness/2]
                    orn = p.getQuaternionFromEuler([0,0,math.pi/2])
                    length = wall_len
                else:  # right
                    pos = [cx + room_size, cy + wall_len/2, wall_thickness/2]
                    orn = p.getQuaternionFromEuler([0,0,math.pi/2])
                    length = wall_len
                if urdf_path:
                    body = p.loadURDF(urdf_path, pos, orn,
                                      globalScaling=length)
                else:
                    col = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_thickness/2])
                    vis = p.createVisualShape(
                        p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_thickness/2],
                        rgbaColor=[0.4,0.4,0.4,1])
                    body = p.createMultiBody(0, col, vis, pos, orn)
                ids.append(body)
    return ids


def _maze_dfs(width: int, height: int, rng: random.Random):
    stack = [(0, 0)]
    visited = [[False]*width for _ in range(height)]
    visited[0][0] = True
    edges = set()
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                neighbors.append((nx, ny))
        if neighbors:
            nx, ny = rng.choice(neighbors)
            edges.add(((x,y),(nx,ny)))
            visited[ny][nx] = True
            stack.append((nx,ny))
        else:
            stack.pop()
    return edges


def create_maze_scene(grid_size: int = 4,
                       passage_width: float = 1.0,
                       wall_thickness: float = 0.1,
                       urdf_path: str = "",
                       seed: int = None) -> List[int]:
    """Generate a simple maze using DFS."""
    rng = random.Random(seed)
    ids = []
    edges = _maze_dfs(grid_size, grid_size, rng)
    half = (grid_size * passage_width) / 2.0
    for x in range(grid_size):
        for y in range(grid_size):
            cx = -half + x * passage_width
            cy = -half + y * passage_width
            # bottom wall
            if ((x,y-1),(x,y)) not in edges and ((x,y),(x,y-1)) not in edges:
                pos = [cx + passage_width/2, cy, wall_thickness/2]
                orn = p.getQuaternionFromEuler([0,0,0])
                length = passage_width
                if urdf_path:
                    body = p.loadURDF(urdf_path, pos, orn, globalScaling=length)
                else:
                    col = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_thickness/2])
                    vis = p.createVisualShape(
                        p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_thickness/2],
                        rgbaColor=[0.5,0.5,0.5,1])
                    body = p.createMultiBody(0, col, vis, pos, orn)
                ids.append(body)
            # right wall
            if ((x+1,y),(x,y)) not in edges and ((x,y),(x+1,y)) not in edges:
                pos = [cx + passage_width, cy + passage_width/2, wall_thickness/2]
                orn = p.getQuaternionFromEuler([0,0,math.pi/2])
                length = passage_width
                if urdf_path:
                    body = p.loadURDF(urdf_path, pos, orn, globalScaling=length)
                else:
                    col = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_thickness/2])
                    vis = p.createVisualShape(
                        p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_thickness/2],
                        rgbaColor=[0.5,0.5,0.5,1])
                    body = p.createMultiBody(0, col, vis, pos, orn)
                ids.append(body)
    # outer boundary
    size = grid_size * passage_width
    boundary = create_room_scene(1, size, wall_thickness, urdf_path=urdf_path, seed=seed)
    ids.extend(boundary)
    return ids
