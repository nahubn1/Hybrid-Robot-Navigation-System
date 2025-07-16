from pathlib import Path
import sys

import torch

SRC_PATH = Path(__file__).resolve().parents[3] / 'src'
sys.path.append(str(SRC_PATH))

from dnn_guidance.modules import FiLMLayer


def test_film_layer_shape_and_op():
    B, C, H, W = 2, 256, 8, 8
    layer = FiLMLayer(robot_param_dim=2, feature_map_channels=C)
    feature_map = torch.randn(B, C, H, W)
    robot_vec = torch.randn(B, 2)
    out = layer(feature_map, robot_vec)
    assert out.shape == feature_map.shape
    assert not torch.allclose(out, feature_map)
