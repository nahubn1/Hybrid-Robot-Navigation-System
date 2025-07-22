from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List
import random

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RobotParamScaling:
    clearance: Tuple[float, float] = (0.0, 5.0)
    step_size: Tuple[float, float] = (0.0, 40.0)

    def scale(self, clearance: float, step: float) -> Tuple[float, float]:
        cl_min, cl_max = self.clearance
        st_min, st_max = self.step_size
        cl_scaled = 2 * (clearance - cl_min) / (cl_max - cl_min) - 1
        st_scaled = 2 * (step - st_min) / (st_max - st_min) - 1
        return float(cl_scaled), float(st_scaled)


def _pair_files(samples_dir: Path, gt_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for sample in sorted(samples_dir.glob("*.npz")):
        gt = gt_dir / sample.name
        if gt.exists():
            pairs.append((sample, gt))
    return pairs


class PathfindingDataset(Dataset):
    """Dataset for pathfinding training samples and heatmaps."""

    def __init__(
        self,
        samples_dir: str | Path,
        ground_truth_dir: str | Path,
        *,
        augment: bool = False,
        scaling: RobotParamScaling | None = None,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.gt_dir = Path(ground_truth_dir)
        self.pairs = _pair_files(self.samples_dir, self.gt_dir)
        if not self.pairs:
            raise ValueError("No paired .npz files found")
        self.augment = augment
        self.scaling = scaling or RobotParamScaling()

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sample_path, gt_path = self.pairs[idx]
        with np.load(sample_path) as data:
            grid = data["map"].astype(np.uint8)
            clearance = float(data["clearance"])
            step_size = float(data["step_size"])
        with np.load(gt_path) as gtd:
            heatmap = gtd["heatmap"].astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                grid = np.fliplr(grid).copy()
                heatmap = np.fliplr(heatmap).copy()
            if random.random() < 0.5:
                grid = np.flipud(grid).copy()
                heatmap = np.flipud(heatmap).copy()

        start = (grid == 8).astype(np.float32)
        goal = (grid == 9).astype(np.float32)
        traversable = ((grid == 0) | (grid == 8) | (grid == 9)).astype(np.float32)
        obstacles = 1.0 - traversable
        grid_tensor = torch.from_numpy(np.stack([start, goal, traversable, obstacles]))

        cl_scaled, step_scaled = self.scaling.scale(clearance, step_size)
        robot_tensor = torch.tensor([cl_scaled, step_scaled], dtype=torch.float32)

        heatmap_tensor = torch.from_numpy(np.ascontiguousarray(heatmap[None, ...]))

        return (grid_tensor.float(), robot_tensor), heatmap_tensor.float()

    def get_raw_item(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the raw arrays without any preprocessing or augmentation."""
        sample_path, gt_path = self.pairs[idx]
        with np.load(sample_path) as data:
            grid = data["map"].astype(np.uint8)
            clearance = float(data["clearance"])
            step_size = float(data["step_size"])
        with np.load(gt_path) as gtd:
            heatmap = gtd["heatmap"].astype(np.float32)

        robot = np.array([clearance, step_size], dtype=np.float32)
        return grid, robot, heatmap
