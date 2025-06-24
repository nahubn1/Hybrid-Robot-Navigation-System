#!/usr/bin/env python3
"""Generate ground truth heatmaps from occupancy grid files."""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from data_generation.ground_truth_heatmaps import (
    PRMConfig,
    HeatmapConfig,
    generate_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate utility heatmaps")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--method", type=str, choices=["betweenness", "component"], default="betweenness")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--overlay", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_overlay(grid: np.ndarray, heatmap: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.imshow(grid, cmap="gray", origin="lower")
    plt.imshow(heatmap, cmap="hot", origin="lower", alpha=0.5)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def process_file(file_path: Path, out_dir: Path, prm_cfg: PRMConfig, hm_cfg: HeatmapConfig, method: str, overlay: bool) -> None:
    grid = np.load(file_path)
    heatmap = generate_heatmap(grid, prm_cfg, hm_cfg, method)
    out_file = out_dir / file_path.name.replace("env_", "gt_")
    np.save(out_file, heatmap)
    if overlay:
        img_file = out_file.with_suffix(".png")
        save_overlay(grid, heatmap, img_file)


def main() -> None:
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(Path(args.config))
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prm_cfg = PRMConfig(num_samples=cfg.get("samples", args.samples), radius=cfg.get("radius", args.radius))
    hm_cfg = HeatmapConfig(sigma=cfg.get("sigma", args.sigma), top_k=cfg.get("top_k", args.top_k))
    method = cfg.get("method", args.method)

    files = sorted(input_dir.glob("*.npy"))
    worker = partial(process_file, out_dir=output_dir, prm_cfg=prm_cfg, hm_cfg=hm_cfg, method=method, overlay=args.overlay)
    if args.processes > 1:
        with Pool(args.processes) as pool:
            list(pool.imap_unordered(worker, files))
    else:
        for f in files:
            worker(f)


if __name__ == "__main__":
    main()
