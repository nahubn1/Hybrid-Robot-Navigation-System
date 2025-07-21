"""High-level model inference interface."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .data_loader import RobotParamScaling


class InferenceHandler:
    """Run inference with a trained model using simple NumPy inputs.

    Parameters
    ----------
    model_class : type
        Class object of the neural network architecture to instantiate.
    model_checkpoint_path : str or Path or None
        Path to a ``state_dict`` checkpoint for the model. If ``None`` the model
        is used with randomly initialised weights.
    device : str or torch.device
        Device on which to run inference.
    """

    def __init__(self, model_class: type, model_checkpoint_path: str | Path | None, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model_class()
        if model_checkpoint_path is not None:
            state_dict = torch.load(model_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.scaling = RobotParamScaling()

    def _preprocess(self, grid_numpy: np.ndarray, robot_numpy: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw NumPy arrays to batched tensors on the target device."""
        grid = np.asarray(grid_numpy)
        robot = np.asarray(robot_numpy)

        start = (grid == 8).astype(np.float32)
        goal = (grid == 9).astype(np.float32)
        traversable = ((grid == 0) | (grid == 8) | (grid == 9)).astype(np.float32)
        obstacles = 1.0 - traversable
        grid_tensor = torch.from_numpy(np.stack([start, goal, traversable, obstacles])).unsqueeze(0)

        cl_scaled, step_scaled = self.scaling.scale(float(robot[0]), float(robot[1]))
        robot_tensor = torch.tensor([[cl_scaled, step_scaled]], dtype=torch.float32)

        return grid_tensor.to(self.device), robot_tensor.to(self.device)

    def predict(self, grid_numpy: np.ndarray, robot_numpy: np.ndarray) -> np.ndarray:
        """Run the model's forward pass and return a heatmap as a NumPy array."""
        grid_tensor, robot_tensor = self._preprocess(grid_numpy, robot_numpy)
        with torch.no_grad():
            logits = self.model(grid_tensor, robot_tensor)
            probs = torch.sigmoid(logits)
        heatmap = probs.squeeze(0).squeeze(0).cpu().numpy()
        return heatmap
