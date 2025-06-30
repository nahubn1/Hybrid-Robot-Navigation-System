#!/usr/bin/env python3
"""Generate training samples using a YAML configuration."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import random

import numpy as np
import yaml

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
import sys
sys.path.append(str(SRC_PATH))

from data_generation.office_map_generator import (
    generate_office_map,
    flood_fill_connected,
)
from scipy import ndimage


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_config(path: Path) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def choose_free_cell(grid: np.ndarray, rng: random.Random, clearance: float = 0.0) -> tuple[int, int]:
    free_mask = grid == 0
    if clearance > 0:
        distance = ndimage.distance_transform_edt(free_mask)
        valid = np.argwhere(distance >= clearance)
    else:
        valid = np.argwhere(free_mask)
    if len(valid) == 0:
        raise ValueError("No valid free cell found with required clearance")
    idx = rng.randint(0, len(valid) - 1)
    y, x = valid[idx]
    return int(y), int(x)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate training data')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg.get('seed', 0))
    rng = random.Random(seed)
    np.random.seed(seed)

    out_dir = Path(cfg.get('output_dir', 'data/training_samples'))
    out_dir.mkdir(parents=True, exist_ok=True)

    num_maps = int(cfg.get('num_maps', 1))
    samples_per_map = int(cfg.get('samples_per_map', 1))
    robots: List[Dict] = cfg.get('robots', [{'clearance': 0.2, 'step_size': 1.0}])

    for map_idx in range(num_maps):
        logging.info('Generating map %d/%d', map_idx + 1, num_maps)
        base = generate_office_map(cfg, rng)
        if not flood_fill_connected(base):
            logging.warning('Map %d is not fully connected', map_idx)
        for sample_idx in range(samples_per_map):
            for robot_idx, robot in enumerate(robots):
                sample = base.copy()
                clearance = float(robot.get('clearance', 0.0))
                sy, sx = choose_free_cell(sample, rng, clearance)
                gy, gx = choose_free_cell(sample, rng, clearance)
                while (gy, gx) == (sy, sx):
                    gy, gx = choose_free_cell(sample, rng, clearance)
                sample[sy, sx] = 8  # Use 8 for start
                sample[gy, gx] = 9  # Use 9 for goal
                name = f'map{map_idx:04d}_s{sample_idx:02d}_r{robot_idx:02d}.npz'
                out_path = out_dir / name
                np.savez_compressed(
                    out_path,
                    map=sample,
                    clearance=float(robot.get('clearance', 0.2)),
                    step_size=float(robot.get('step_size', 1.0)),
                    config=json.dumps(cfg),
                )


if __name__ == '__main__':
    main()
