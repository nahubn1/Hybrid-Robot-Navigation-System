#!/usr/bin/env python3
"""Compose a layered visualization of ground truth data for one sample."""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
import yaml


import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Use a key that doesn't conflict with matplotlib's shortcuts, e.g., 'N' for next
SWITCH_KEY = 'n'


def grid_hash(grid: np.ndarray) -> str:
    """Return a short hash string for ``grid``."""
    h = hashlib.sha256()
    h.update(grid.tobytes())
    return h.hexdigest()[:16]


def compose_visualization(
    sample_path: Path,
    gt_dir: Path,
    *,
    show_indices: bool,
    show_prm: bool,
    prm_samples: int,
    prm_k: int,
    filtered_cache_dir: Path,
) -> plt.Figure:
    """Return a matplotlib figure visualizing a ground truth sample.

    The PRM overlay is loaded from ``filtered_cache_dir`` using the same cache key
    employed during ground truth generation.
    """
    base = gt_dir / sample_path.with_suffix("").name

    # Load base sample and ground truth arrays
    sample = np.load(sample_path, allow_pickle=True)
    grid = sample["map"]
    gt = np.load(base.with_suffix(".npz"))
    indices = gt["indices"]
    mask = gt["mask"]
    heat = gt["heatmap"]

    # Determine start and goal coordinates (x, y)
    start_pos = np.argwhere(grid == 8)
    goal_pos = np.argwhere(grid == 9)
    if start_pos.size == 0 or goal_pos.size == 0:
        raise ValueError("Sample does not contain start/goal markers")
    start = start_pos[0][::-1]
    goal = goal_pos[0][::-1]

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- Layer 1: base map ---
    free_color = np.ones((*grid.shape, 3)) * 0.9  # light grey
    obstacles = (grid != 0) & (grid != 8) & (grid != 9)
    free_color[obstacles] = 0.3  # dark grey
    ax.imshow(free_color, origin="lower")

    # --- Layer 2: start/goal markers ---
    start_plot = ax.scatter(start[0], start[1], s=120, c="green", marker="o",
                            edgecolors="black", linewidths=1, label="Start", zorder=3)
    goal_plot = ax.scatter(goal[0], goal[1], s=150, c="red", marker="*",
                           edgecolors="black", linewidths=1, label="Goal", zorder=3)

    # --- Optional Layer 3: PRM roadmap ---
    if show_prm:
        clearance = float(sample["clearance"])
        step = float(sample["step_size"])
        key = f"{grid_hash(grid)}_{prm_samples}_{prm_k}_{clearance}_{step}"

        prm_path = filtered_cache_dir / f"{key}_filtered_prm.pkl"
        if prm_path.exists():
            with open(prm_path, "rb") as f:
                prm = pickle.load(f)
            for u, v in prm.edges():
                x1, y1 = prm.nodes[u]["pos"]
                x2, y2 = prm.nodes[v]["pos"]
                ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.5, alpha=0.5, zorder=2)
            if prm.number_of_nodes() > 0:
                pts = np.array([prm.nodes[n]["pos"] for n in prm.nodes])
                ax.scatter(pts[:, 0], pts[:, 1], s=8, c="black", alpha=0.7, zorder=2)
        else:
            ax.text(0.5, 0.5, "PRM cache missing", transform=ax.transAxes, ha="center", va="center",
                    color="red", fontsize=10, zorder=5)

    # --- Layer 4: probabilistic heatmap ---
    ax.imshow(heat, cmap="viridis", alpha=0.6, origin="lower")

    # --- Layer 5: raw planner path ---
    path_coords = np.argwhere(indices > 0)
    order = np.argsort(indices[path_coords[:, 0], path_coords[:, 1]])
    path = path_coords[order]
    (path_plot,) = ax.plot(path[:, 1], path[:, 0], color="cyan", linewidth=1,
                           label="Raw Path", zorder=4)

    # --- Optional Layer 6: step indices ---
    if show_indices:
        for y, x in path:
            idx = int(indices[y, x])
            ax.text(x, y, str(idx), fontsize=6, color="black",
                    ha="center", va="center", zorder=5)

    ax.set_title(f"Ground Truth: {sample_path.stem}")
    ax.axis("off")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ground truth samples")
    default_cfg = (
        Path(__file__).resolve().parents[2]
        / "configs/data_generation/visualize_ground_truth.yaml"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="YAML configuration file",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gt_dir = Path(cfg.get("ground_truth_dir", "data/ground_truth"))
    show_indices = bool(cfg.get("show_indices", False))
    show_prm = bool(cfg.get("show_prm", True))
    prm_samples = int(cfg.get("samples", 500))
    prm_k = int(cfg.get("k_neighbors", 10))
    filtered_cache_dir = Path(cfg.get("filtered_cache_dir", ".cache/filtered"))
    root = tk.Tk()
    root.withdraw()
    while True:
        sample_path = filedialog.askopenfilename(
            title="Select .npz sample file", filetypes=[("NPZ files", "*.npz")]
        )
        if not sample_path:
            break
        fig = compose_visualization(
            Path(sample_path),
            gt_dir,
            show_indices=show_indices,
            show_prm=show_prm,
            prm_samples=prm_samples,
            prm_k=prm_k,
            filtered_cache_dir=filtered_cache_dir,
        )
        switch_file = {'next': False}

        def on_key(event):
            if event.key == SWITCH_KEY:
                switch_file['next'] = True
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        if not switch_file['next']:
            break


if __name__ == "__main__":
    main()
