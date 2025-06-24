import numpy as np
import pybullet as p


def generate_occupancy_grid(area_size: float = 5.0,
                            resolution: int = 128,
                            height: float = 5.0) -> np.ndarray:
    """Generate a top-down occupancy grid via batched ray casting."""
    half = area_size / 2.0
    cell = area_size / resolution
    ray_from = []
    ray_to = []
    for i in range(resolution):
        y = -half + (i + 0.5) * cell
        for j in range(resolution):
            x = -half + (j + 0.5) * cell
            ray_from.append([x, y, height])
            ray_to.append([x, y, -height])
    results = p.rayTestBatch(ray_from, ray_to)
    occ = np.zeros((resolution, resolution), dtype=np.uint8)
    idx = 0
    for i in range(resolution):
        for j in range(resolution):
            hit = results[idx][0] != -1
            occ[i, j] = 1 if hit else 0
            idx += 1
    return occ
