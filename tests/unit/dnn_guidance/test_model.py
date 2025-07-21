from pathlib import Path
import sys

import torch

SRC_PATH = Path(__file__).resolve().parents[3] / 'src'
sys.path.append(str(SRC_PATH))

from dnn_guidance.model import UNetFiLM, HRFiLMNet
from dnn_guidance.config import UNetConfig, HRFiLMConfig


def test_unet_film_forward_pass():
    model = UNetFiLM()
    model.eval()
    B = 2
    grid_tensor = torch.randn(B, 4, 200, 200)
    robot_tensor = torch.randn(B, 2)
    logits = model(grid_tensor, robot_tensor)
    assert logits.shape == (B, 1, 200, 200)


def test_unet_config_loading(tmp_path):
    cfg_text = (
        "in_channels: 4\n"
        "enc_channels: [16, 32, 64]\n"
        "bottleneck_channels: 128\n"
        "robot_param_dim: 2\n"
        "dec_channels: [64, 32, 16]\n"
        "out_channels: 1\n"
    )
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(cfg_text)
    cfg = UNetConfig.from_yaml(cfg_path)
    model = UNetFiLM(cfg)
    grid = torch.randn(1, cfg.in_channels, 200, 200)
    robot = torch.randn(1, cfg.robot_param_dim)
    out = model(grid, robot)
    assert out.shape == (1, cfg.out_channels, 200, 200)


def test_hrfilmnet_forward_pass():
    model = HRFiLMNet()
    model.eval()
    B = 2
    grid = torch.randn(B, 4, 200, 200)
    robot = torch.randn(B, 2)
    out = model(grid, robot)
    assert out.shape == (B, 1, 200, 200)


def test_hrfilm_config_loading(tmp_path):
    cfg_text = (
        "in_channels: 4\n"
        "robot_param_dim: 2\n"
        "stage_channels: [32, 64, 128, 256]\n"
        "out_channels: 1\n"
    )
    path = tmp_path / "hr_cfg.yaml"
    path.write_text(cfg_text)
    cfg = HRFiLMConfig.from_yaml(path)
    model = HRFiLMNet(cfg)
    grid = torch.randn(1, cfg.in_channels, 200, 200)
    robot = torch.randn(1, cfg.robot_param_dim)
    out = model(grid, robot)
    assert out.shape == (1, cfg.out_channels, 200, 200)

