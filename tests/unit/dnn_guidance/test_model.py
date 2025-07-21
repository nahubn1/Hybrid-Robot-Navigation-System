import sys
from pathlib import Path

import torch

SRC_PATH = Path(__file__).resolve().parents[3] / "src"
sys.path.append(str(SRC_PATH))

from dnn_guidance.config import (
    DiffusionUNetConfig,
    HRFiLMConfig,
    ResNetFPNFiLMConfig,
    UNetConfig,
)
from dnn_guidance.model import (
    ConditionalDenoisingUNet,
    HRFiLMNet,
    ResNetFPNFiLM,
    UNetFiLM,
)


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


def test_resnet_fpn_film_forward_and_config(tmp_path):
    cfg_text = "in_channels: 4\n" "robot_param_dim: 2\n" "out_channels: 1\n"
    path = tmp_path / "rff_cfg.yaml"
    path.write_text(cfg_text)
    cfg = ResNetFPNFiLMConfig.from_yaml(path)
    model = ResNetFPNFiLM(cfg)
    B = 2
    grid = torch.randn(B, cfg.in_channels, 200, 200)
    robot = torch.randn(B, cfg.robot_param_dim)
    out = model(grid, robot)
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.head.modules())
    assert out.shape == (B, cfg.out_channels, 200, 200)


def test_diffusion_unet_forward_and_config(tmp_path):
    cfg_text = (
        "in_channels: 1\n"
        "grid_channels: 4\n"
        "robot_param_dim: 2\n"
        "time_dim: 16\n"
        "enc_channels: [8, 16, 32]\n"
        "bottleneck_channels: 64\n"
        "dec_channels: [32, 16, 8]\n"
    )
    path = tmp_path / "diff_cfg.yaml"
    path.write_text(cfg_text)
    cfg = DiffusionUNetConfig.from_yaml(path)
    model = ConditionalDenoisingUNet(
        in_channels=cfg.in_channels,
        grid_channels=cfg.grid_channels,
        robot_param_dim=cfg.robot_param_dim,
        time_dim=cfg.time_dim,
        enc_channels=cfg.enc_channels,
        bottleneck_channels=cfg.bottleneck_channels,
        dec_channels=cfg.dec_channels,
    )
    B = 2
    noisy = torch.randn(B, cfg.in_channels, 200, 200)
    t = torch.randint(0, 10, (B,))
    grid = torch.randn(B, cfg.grid_channels, 200, 200)
    robot = torch.randn(B, cfg.robot_param_dim)
    out = model(noisy, t, grid, robot)
    assert out.shape == (B, 1, 200, 200)
