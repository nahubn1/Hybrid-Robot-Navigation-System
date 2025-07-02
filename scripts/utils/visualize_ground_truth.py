#!/usr/bin/env python3
"""Compose a layered visualization of ground truth data for one sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Use a key that doesn't conflict with matplotlib's shortcuts, e.g., 'N' for next
SWITCH_KEY = 'n'


def compose_visualization(sample_path: Path, gt_dir: Path, show_indices: bool = False) -> plt.Figure:
    """Return a matplotlib figure visualizing a ground truth sample."""
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

    # --- Layer 3: probabilistic heatmap ---
    ax.imshow(heat, cmap="viridis", alpha=0.6, origin="lower")

    # --- Layer 4: raw planner path ---
    path_coords = np.argwhere(indices > 0)
    order = np.argsort(indices[path_coords[:, 0], path_coords[:, 1]])
    path = path_coords[order]
    (path_plot,) = ax.plot(path[:, 1], path[:, 0], color="cyan", linewidth=1,
                           label="Raw Path", zorder=4)

    # --- Optional Layer 5: step indices ---
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
    parser.add_argument("--ground_truth_dir", type=Path, required=True,
                        help="Directory containing ground truth .npy files")
    parser.add_argument("--show-indices", action="store_true",
                        help="Overlay step numbers on the path")
    args = parser.parse_args()

    root = tk.Tk()
    root.withdraw()
    while True:
        sample_path = filedialog.askopenfilename(title="Select .npz sample file", filetypes=[("NPZ files", "*.npz")])
        if not sample_path:
            break
        fig = compose_visualization(Path(sample_path), args.ground_truth_dir, args.show_indices)
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
