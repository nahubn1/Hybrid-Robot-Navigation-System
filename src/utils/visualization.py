import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Mapping, Any


def plot_inference_comparison(data: Mapping[str, Any], *, output_path: str | Path | None = None) -> plt.Figure:
    """Generate a multi-panel comparison plot for inference results.

    Parameters
    ----------
    data : Mapping[str, Any]
        Dictionary containing ``'input_grid'`` (``np.ndarray``),
        ``'ground_truth_heatmap'`` (``np.ndarray``) and ``'predictions'``,
        which itself is a mapping of model name to predicted heatmap.
    output_path : str or Path, optional
        If provided, the figure is saved to this location.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure instance.
    """
    grid = np.squeeze(np.asarray(data['input_grid']))
    gt_heatmap = np.squeeze(np.asarray(data['ground_truth_heatmap']))
    predictions: Mapping[str, np.ndarray] = data.get('predictions', {})

    ncols = 2 + len(predictions)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    axes = np.atleast_1d(axes)

    # Subplot 1: input grid
    cmap_grid = ListedColormap(['white', 'black', 'green', 'red'])
    grid_vis = np.zeros_like(grid, dtype=int)
    grid_vis[(grid != 0) & (grid != 8) & (grid != 9)] = 1
    grid_vis[grid == 8] = 2
    grid_vis[grid == 9] = 3
    axes[0].imshow(grid_vis, cmap=cmap_grid, origin='lower')
    axes[0].set_title('Input Problem')

    # Determine start/goal for scatter overlays
    start = np.argwhere(grid == 8)
    goal = np.argwhere(grid == 9)

    # Ground truth heatmap with map overlay
    axes[1].imshow(grid_vis, cmap=cmap_grid, origin='lower')
    axes[1].imshow(gt_heatmap, cmap='viridis', vmin=0, vmax=1, origin='lower', alpha=0.6)
    if start.size:
        axes[1].scatter(start[:, 1], start[:, 0], c='green', edgecolors='black', s=50)
    if goal.size:
        axes[1].scatter(goal[:, 1], goal[:, 0], c='red', edgecolors='black', s=50, marker='*')
    axes[1].set_title('Ground Truth')

    # Predictions with map overlay
    for ax, (name, heatmap) in zip(axes[2:], predictions.items()):
        hm = np.squeeze(np.asarray(heatmap))
        ax.imshow(grid_vis, cmap=cmap_grid, origin='lower')
        ax.imshow(hm, cmap='viridis', vmin=0, vmax=1, origin='lower', alpha=0.6)
        if start.size:
            ax.scatter(start[:, 1], start[:, 0], c='green', edgecolors='black', s=50)
        if goal.size:
            ax.scatter(goal[:, 1], goal[:, 0], c='red', edgecolors='black', s=50, marker='*')
        ax.set_title(name)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
    return fig
