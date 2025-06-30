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
from typing import Tuple

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from planning_algorithms.prm import build_prm
from planning_algorithms.dstar_lite import DStarLite
from utils.image_processing import distance_transform, dilate, gaussian_blur
from utils.graph_helpers import filter_graph, snap_point


CACHE_DIR = Path('.cache')
CACHE_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate ground truth data')
    p.add_argument('--input-dir', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--k-neighbors', type=int, default=10)
    p.add_argument('--processes', type=int, default=1)
    p.add_argument('--dilate-radius', type=int, default=2)
    p.add_argument('--blur-sigma', type=float, default=1.0)
    return p.parse_args()


def load_npz(file_path: Path):
    data = np.load(file_path, allow_pickle=True)
    grid = data['map']
    c = float(data['clearance'])
    s = float(data['step_size'])
    return grid, c, s


def preprocess_map(map_id: str, grid: np.ndarray, num_samples: int, k: int):
    dist_path = CACHE_DIR / f'{map_id}_dist.npy'
    prm_path = CACHE_DIR / f'{map_id}_prm.pkl'
    if dist_path.exists() and prm_path.exists():
        dist = np.load(dist_path)
        with open(prm_path, 'rb') as f:
            prm = pickle.load(f)
        return dist, prm
    dist = distance_transform((grid != 0).astype(np.uint8))
    prm = build_prm((grid != 0).astype(np.uint8), num_samples=num_samples, k=k)
    np.save(dist_path, dist)
    with open(prm_path, 'wb') as f:
        pickle.dump(prm, f)
    return dist, prm


def densify_path(nodes: list[Tuple[int, int]], step: float) -> list[Tuple[int, int]]:
    result = []
    for i in range(len(nodes) - 1):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i + 1]
        dist = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(math.ceil(dist / step)))
        for j in range(n):
            t = j / n
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            result.append((x, y))
    result.append(nodes[-1])
    return result


def process_file(file_path: Path, output_dir: Path, samples: int, k: int, dil_rad: int, blur_sigma: float):
    grid, clearance, step = load_npz(file_path)
    map_id = file_path.stem.split('_')[0]
    start = tuple(np.argwhere(grid == 8)[0][::-1])
    goal = tuple(np.argwhere(grid == 9)[0][::-1])
    dist, base_prm = preprocess_map(map_id, grid, samples, k)
    filtered = filter_graph(base_prm, dist, clearance, step)
    if filtered.number_of_nodes() == 0:
        return
    goal_node = snap_point(filtered, goal)
    planner = DStarLite(filtered, goal_node)
    start_node = snap_point(filtered, start)
    node_path = planner.replan(start_node)
    if not node_path:
        return
    coord_path = [filtered.nodes[n]['pos'] for n in node_path]
    dense_path = densify_path(coord_path, step)
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
    np.save(out_base.with_suffix('.indices.npy'), indices)
    np.save(out_base.with_suffix('.mask.npy'), mask)
    np.save(out_base.with_suffix('.heat.npy'), heat)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input_dir).glob('*.npz'))
    worker = partial(process_file, output_dir=out_dir, samples=args.samples, k=args.k_neighbors, dil_rad=args.dilate_radius, blur_sigma=args.blur_sigma)
    if args.processes > 1:
        with Pool(args.processes) as pool:
            list(pool.imap_unordered(worker, files))
    else:
        for f in files:
            worker(f)


if __name__ == '__main__':
    main()
