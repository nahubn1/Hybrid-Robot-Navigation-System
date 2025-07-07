#!/usr/bin/env python3
"""Generate ground truth paths and heatmaps using PRM and D* Lite."""

from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import pickle
import sys
import math
from typing import Dict, Optional, Tuple
from collections import deque
import random
import warnings
import hashlib
import shutil
from tqdm import tqdm
import networkx as nx

import numpy as np
import yaml


class GroundTruthGenerationError(RuntimeError):
    """Raised when ground truth generation fails."""

    def __init__(
        self,
        message: str,
        segment: tuple[tuple[int, int], tuple[int, int]] | None = None,
    ) -> None:
        super().__init__(message)
        self.segment = segment


SRC_PATH = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC_PATH))

from planning_algorithms.prm import build_prm
from planning_algorithms.dstar_lite import DStarLite
from utils.image_processing import distance_transform, dilate, gaussian_blur
from utils.graph_helpers import filter_graph, snap_point, bresenham_line



def clear_cache(cache_dir: Path, filtered_cache_dir: Path) -> None:
    """Remove all cached files in ``cache_dir`` and ``filtered_cache_dir``."""
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(filtered_cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filtered_cache_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate ground truth data')
    default_cfg = (
        Path(__file__).resolve().parents[2]
        / 'configs/data_generation/ground_truth_generation.yaml'
    )
    p.add_argument(
        '--config',
        type=str,
        default=str(default_cfg),
        help='YAML configuration file',
    )
    return p.parse_args()


def load_npz(file_path: Path):
    """Load a training sample from an ``npz`` file with validation."""
    if not file_path.exists():
        raise GroundTruthGenerationError(f"load_npz: file not found -> {file_path}")
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as exc:
        raise GroundTruthGenerationError(
            f"load_npz: failed to read {file_path}: {exc}"
        ) from exc

    required = {"map", "clearance", "step_size"}
    missing = required.difference(data.files)
    if missing:
        raise GroundTruthGenerationError(
            f"load_npz: missing keys {missing} in {file_path}"
        )

    grid = data["map"]
    c = float(data["clearance"])
    s = float(data["step_size"])
    cfg = data.get("config", None)
    return grid, c, s, cfg


def preprocess_map(
    map_id: str,
    grid: np.ndarray,
    num_samples: int,
    k: int,
    cache_dir: Path,
):
    """Compute and cache distance transform and PRM for a map.

    ``map_id`` identifies the underlying base map so identical maps across
    different environments reuse the same cached data.
    """
    if grid.size == 0:
        raise GroundTruthGenerationError(
            f"preprocess_map: empty grid for map_id={map_id}"
        )

    key = map_id

    dist_path = cache_dir / f"{key}_dist.npy"
    prm_path = cache_dir / f"{key}_prm.pkl"
    if dist_path.exists() and prm_path.exists():
        try:
            dist = np.load(dist_path)
            with open(prm_path, "rb") as f:
                prm = pickle.load(f)
            return dist, prm
        except Exception:
            # Corrupted cache files can occur if a previous run was interrupted
            # while writing. Remove them and regenerate from scratch.
            dist_path.unlink(missing_ok=True)
            prm_path.unlink(missing_ok=True)
    try:
        dist = distance_transform((grid != 0).astype(np.uint8))
        prm = build_prm((grid != 0).astype(np.uint8), num_samples=num_samples, k=k)
    except Exception as exc:
        raise GroundTruthGenerationError(
            f"preprocess_map: failed to build PRM for {map_id}: {exc}"
        ) from exc
    try:
        np.save(dist_path, dist)
        with open(prm_path, "wb") as f:
            pickle.dump(prm, f)
    except Exception as exc:
        raise GroundTruthGenerationError(
            f"preprocess_map: failed to write cache for {map_id}: {exc}"
        ) from exc
    return dist, prm


def densify_path(nodes: list[Tuple[int, int]], step: float) -> list[Tuple[int, int]]:
    """Return all grid cells along ``nodes`` using Bresenham line expansion."""
    if step <= 0:
        raise GroundTruthGenerationError("densify_path: step must be positive")
    if len(nodes) < 2:
        raise GroundTruthGenerationError("densify_path: need at least two nodes")
    result: list[Tuple[int, int]] = []
    last: Tuple[int, int] | None = None
    for (x1, y1), (x2, y2) in zip(nodes[:-1], nodes[1:]):
        for x, y in bresenham_line(x1, y1, x2, y2):
            pt = (int(x), int(y))
            if pt == last:
                continue
            result.append(pt)
            last = pt
    if result[-1] != nodes[-1]:
        result.append(nodes[-1])
    return result


def path_collision_free(
    grid: np.ndarray, path: list[Tuple[int, int]]
) -> tuple[bool, tuple[tuple[int, int], tuple[int, int]] | None]:
    """Check that ``path`` does not intersect occupied cells in ``grid``.

    Returns a tuple ``(collision_free, segment)`` where ``segment`` is the first
    path segment that collides with an obstacle, or ``None`` if the path is
    collision-free.
    """
    occ = (grid != 0).astype(np.uint8)
    occ[grid == 8] = 0
    occ[grid == 9] = 0
    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
        for x, y in bresenham_line(x1, y1, x2, y2):
            if occ[int(y), int(x)]:
                return False, ((int(x1), int(y1)), (int(x2), int(y2)))
    return True, None


def grid_shortest_path(
    grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
) -> list[Tuple[int, int]]:
    """Return a simple BFS path on the grid from ``start`` to ``goal``."""
    h, w = grid.shape
    sy, sx = start[1], start[0]
    gy, gx = goal[1], goal[0]
    q = deque([(sy, sx)])
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sy, sx): None}
    free = lambda y, x: grid[y, x] in (0, 8, 9)
    while q:
        y, x = q.popleft()
        if (y, x) == (gy, gx):
            break
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and free(ny, nx):
                if (ny, nx) not in parents:
                    parents[(ny, nx)] = (y, x)
                    q.append((ny, nx))
    if (gy, gx) not in parents:
        return []
    path = []
    cur = (gy, gx)
    while cur is not None:
        cy, cx = cur
        path.append((cx, cy))
        cur = parents[cur]
    path.reverse()
    return path


def _plan(graph: nx.Graph, start: Tuple[int, int], goal: Tuple[int, int]) -> list[int]:
    """Attempt to plan a path on ``graph`` from ``start`` to ``goal``."""
    try:
        goal_node = snap_point(graph, goal)
        planner = DStarLite(graph, goal_node)
        start_node = snap_point(graph, start)
        node_path = planner.replan(start_node)
    except Exception:
        return []
    return node_path or []


def reposition_start_goal(
    grid: np.ndarray, graph: nx.Graph
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Choose two random nodes from the largest connected component and update the grid."""
    components = list(nx.connected_components(graph))
    if not components:
        raise GroundTruthGenerationError(
            "reposition_start_goal: graph has no components"
        )
    largest = max(components, key=len)
    if len(largest) < 2:
        raise GroundTruthGenerationError("reposition_start_goal: component too small")
    s_node, g_node = random.sample(list(largest), 2)
    sx, sy = graph.nodes[s_node]["pos"]
    gx, gy = graph.nodes[g_node]["pos"]
    grid[grid == 8] = 0
    grid[grid == 9] = 0
    if 0 <= sy < grid.shape[0] and 0 <= sx < grid.shape[1]:
        grid[int(sy), int(sx)] = 8
    if 0 <= gy < grid.shape[0] and 0 <= gx < grid.shape[1]:
        grid[int(gy), int(gx)] = 9
    return (int(sx), int(sy)), (int(gx), int(gy))


def process_file(
    file_path: Path,
    output_dir: Path,
    samples: int,
    k: int,
    dil_rad: int,
    blur_sigma: float,
    cache_dir: Path,
    filtered_cache_dir: Path,
    save_filtered_prm: bool,
):
    grid, clearance, step, cfg = load_npz(file_path)
    map_id = file_path.stem.split("_")[0]
    starts = np.argwhere(grid == 8)
    goals = np.argwhere(grid == 9)
    if starts.size == 0 or goals.size == 0:
        raise GroundTruthGenerationError(
            f"process_file: start/goal missing in {file_path}"
        )
    start = tuple(starts[0][::-1])
    goal = tuple(goals[0][::-1])

    dist, base_prm = preprocess_map(map_id, grid, samples, k, cache_dir)
    filtered = filter_graph(base_prm, dist, clearance, step)

    cache_key = file_path.stem

    filtered_path = filtered_cache_dir / f"{cache_key}_filtered_prm.pkl"
    if save_filtered_prm and not filtered_path.exists():
        try:
            with open(filtered_path, "wb") as f:
                pickle.dump(filtered, f)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to cache filtered PRM for {file_path}: {exc}", RuntimeWarning
            )
    if filtered.number_of_nodes() == 0:
        raise GroundTruthGenerationError(
            f"process_file: no nodes left after filtering for {file_path}"
        )
    node_path = _plan(filtered, start, goal)
    if len(node_path) < 2:
        try:
            start, goal = reposition_start_goal(grid, filtered)
        except GroundTruthGenerationError:
            raise
        node_path = _plan(filtered, start, goal)
        if len(node_path) < 2:
            raise GroundTruthGenerationError(
                f"process_file: planner returned empty path for {file_path}"
            )
        # update the original npz with new start and goal
        try:
            np.savez_compressed(
                file_path,
                map=grid,
                clearance=clearance,
                step_size=step,
                config=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to update input map {file_path}: {exc}", RuntimeWarning
            )
    # Include the actual start and goal positions in the coordinate path so the
    # resulting dense path begins and ends exactly at these cells.
    coord_path = [start]
    coord_path.extend(filtered.nodes[n]["pos"] for n in node_path)
    coord_path.append(goal)
    dense_path = densify_path(coord_path, step)
    if not path_collision_free(grid, dense_path):
        fallback = grid_shortest_path(grid, start, goal)
        if not fallback or not path_collision_free(grid, fallback):
            raise GroundTruthGenerationError(
                f"process_file: generated path intersects obstacles for {file_path}"
            )
        dense_path = fallback
    indices = np.zeros_like(grid, dtype=np.int32)
    mask = np.zeros_like(grid, dtype=np.uint8)
    for idx, (x, y) in enumerate(dense_path, start=1):
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            indices[y, x] = idx
            mask[y, x] = 1
    dil = dilate(mask, dil_rad)
    heat = gaussian_blur(dil.astype(float), blur_sigma)
    if heat.max() > 0:
        heat /= heat.max()
    out_base = output_dir / file_path.stem
    try:
        np.savez_compressed(
            out_base.with_suffix(".npz"),
            indices=indices,
            mask=mask,
            heatmap=heat,
        )
    except Exception as exc:
        raise GroundTruthGenerationError(
            f"process_file: failed to write outputs for {file_path}: {exc}"
        ) from exc


def safe_process_file(
    file_path: Path,
    output_dir: Path,
    samples: int,
    k: int,
    dil_rad: int,
    blur_sigma: float,
    cache_dir: Path,
    filtered_cache_dir: Path,
    save_filtered_prm: bool,
) -> None:
    """Run ``process_file`` and issue a warning if it fails."""
    try:
        process_file(
            file_path=file_path,
            output_dir=output_dir,
            samples=samples,
            k=k,
            dil_rad=dil_rad,
            blur_sigma=blur_sigma,
            cache_dir=cache_dir,
            filtered_cache_dir=filtered_cache_dir,
            save_filtered_prm=save_filtered_prm,
        )
    except GroundTruthGenerationError as exc:
        msg = str(exc)
        if exc.segment is not None:
            msg += f" (segment {exc.segment})"
        warnings.warn(msg, RuntimeWarning)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Unexpected error processing {file_path}: {exc}", RuntimeWarning)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    cache_dir = Path(cfg.get('cache_dir', '.cache'))
    filtered_cache_dir = Path(cfg.get('filtered_cache_dir', cache_dir))
    save_filtered_prm = bool(cfg.get('save_filtered_prm', True))
    cache_dir.mkdir(parents=True, exist_ok=True)
    filtered_cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get('clear_cache', False):
        clear_cache(cache_dir, filtered_cache_dir)

    try:
        input_dir = Path(cfg['input_dir'])
        output_dir = Path(cfg['output_dir'])
        samples = int(cfg['samples'])
        k_neigh = int(cfg['k_neighbors'])
        processes = int(cfg['processes'])
        dil_rad = int(cfg['dilate_radius'])
        blur_sigma = float(cfg['blur_sigma'])
    except KeyError as exc:
        raise GroundTruthGenerationError(f'missing configuration key: {exc}') from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob('*.npz'))
    if not files:
        raise GroundTruthGenerationError(
            f"main: no .npz files found in {input_dir}")
    worker = partial(
        safe_process_file,
        output_dir=output_dir,
        samples=samples,
        k=k_neigh,
        dil_rad=dil_rad,
        blur_sigma=blur_sigma,
        cache_dir=cache_dir,
        filtered_cache_dir=filtered_cache_dir,
        save_filtered_prm=save_filtered_prm,
    )
    if processes > 1:
        with Pool(processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(worker, files),
                total=len(files),
                desc="Generating ground truth",
            ):
                pass
    else:
        for f in tqdm(files, desc="Generating ground truth"):
            worker(f)


if __name__ == "__main__":
    main()
