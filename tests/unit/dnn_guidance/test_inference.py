import sys
from pathlib import Path

import numpy as np
import torch

SRC_PATH = Path(__file__).resolve().parents[3] / "src"
sys.path.append(str(SRC_PATH))

from dnn_guidance.model import UNetFiLM
from dnn_guidance.inference import InferenceHandler
from dnn_guidance.data_loader import RobotParamScaling


def _dummy_inputs(size: int = 32):
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[0, 0] = 8
    grid[-1, -1] = 9
    robot = np.array([2.5, 20.0], dtype=np.float32)
    return grid, robot


def test_predict_with_checkpoint(tmp_path):
    model = UNetFiLM()
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)

    handler = InferenceHandler(UNetFiLM, ckpt, device="cpu")

    grid, robot = _dummy_inputs()
    pred = handler.predict(grid, robot)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == grid.shape
    assert pred.min() >= 0.0 and pred.max() <= 1.0

    scaling = RobotParamScaling()
    cl, step = scaling.scale(robot[0], robot[1])
    grid_tensor = torch.from_numpy(
        np.stack([
            (grid == 8).astype(np.float32),
            (grid == 9).astype(np.float32),
            ((grid == 0) | (grid == 8) | (grid == 9)).astype(np.float32),
            1.0 - ((grid == 0) | (grid == 8) | (grid == 9)).astype(np.float32),
        ])
    ).unsqueeze(0)
    robot_tensor = torch.tensor([[cl, step]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(grid_tensor, robot_tensor)).squeeze().numpy()
    assert np.allclose(pred, expected)


def test_predict_without_checkpoint():
    handler = InferenceHandler(UNetFiLM, None, device="cpu")
    grid, robot = _dummy_inputs()
    pred = handler.predict(grid, robot)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == grid.shape
    assert pred.min() >= 0.0 and pred.max() <= 1.0
