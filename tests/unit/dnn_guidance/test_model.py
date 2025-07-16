from pathlib import Path
import sys

import torch

SRC_PATH = Path(__file__).resolve().parents[3] / 'src'
sys.path.append(str(SRC_PATH))

from dnn_guidance.model import UNetFiLM


def test_unet_film_forward_pass():
    model = UNetFiLM()
    model.eval()
    B = 2
    grid_tensor = torch.randn(B, 4, 200, 200)
    robot_tensor = torch.randn(B, 2)
    logits = model(grid_tensor, robot_tensor)
    assert logits.shape == (B, 1, 200, 200)
