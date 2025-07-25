"""High-level model inference interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from .data_loader import RobotParamScaling


class InferenceHandler:
    """Run inference with one or more trained models using simple NumPy inputs.

    The handler can operate in three modes depending on ``model_configs``:

    1. **Single-model** – a single configuration dictionary.
    2. **Hybrid-model** – a list of configuration dictionaries to blend outputs.
    3. **Untrained** – configuration with ``checkpoint_path`` set to ``None``.

    Each configuration dictionary must contain the following keys:

    ``model_class``
        Class object of the neural network architecture to instantiate.
    ``checkpoint_path``
        Optional path to a ``state_dict`` checkpoint. If ``None`` the model is
        left untrained.
    ``weight`` (optional)
        A scaling factor (confidence) for this model's output before the 'max'
        operation. Defaults to ``1.0``.
    """

    def __init__(self, model_configs: Dict[str, Any] | Iterable[Dict[str, Any]], device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        if isinstance(model_configs, dict):
            configs: List[Dict[str, Any]] = [model_configs]
        else:
            configs = list(model_configs)

        self.models: List[Tuple[torch.nn.Module, float]] = []
        for cfg in configs:
            cls = cfg["model_class"]
            ckpt = cfg.get("checkpoint_path")
            weight = float(cfg.get("weight", 1.0))

            model = cls()
            if ckpt is not None:
                state_dict = torch.load(Path(ckpt), map_location=self.device)
                # Allow missing keys when loading older checkpoints
                model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self.models.append((model, weight))

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
        """Run the model's forward pass and return a heatmap as a NumPy array.
        
        If multiple models are loaded, their outputs are scaled by their weight
        and then combined by taking the element-wise maximum.
        """
        grid_tensor, robot_tensor = self._preprocess(grid_numpy, robot_numpy)

        # --- MODIFIED: Will store scaled heatmaps ---
        scaled_heatmaps: List[np.ndarray] = []
        weights: List[float] = []
        with torch.no_grad():
            for model, weight in self.models:
                logits = model(grid_tensor, robot_tensor)
                probs = torch.sigmoid(logits)
                heat = probs.squeeze(0).squeeze(0).cpu().numpy()

                # Scale the heatmap by its model's weight before adding to the list
                scaled_heatmaps.append(heat * weight)
                weights.append(weight)

        # If only one model, return its (scaled) heatmap.
        if len(scaled_heatmaps) == 1:
            # Clip to ensure the output is always a valid probability [0, 1]
            return np.clip(scaled_heatmaps[0], 0.0, 1.0)
        
        total_w = float(sum(weights))

        # Blend by weighted average of the scaled heatmaps
        summed = np.sum(scaled_heatmaps, axis=0)
        blended = summed / total_w if total_w > 0 else summed

        # Clip the final result to ensure it remains a valid probability map.
        return np.clip(blended, 0.0, 1.0)
