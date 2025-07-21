from __future__ import annotations

from pathlib import Path
from torch import nn
import torch

__all__ = ["UNetFiLM", "HRFiLMNet", "ConditionalDenoisingUNet", "create_model"]

from .config import UNetConfig, HRFiLMConfig, DiffusionUNetConfig
from .modules import (
    DoubleConv,
    EncoderBlock,
    DecoderBlock,
    DecoderBlockWithGrid,
    FiLMLayer,
    FiLM,
    ResidualBlock,
    DilatedResidualBlock,
)
from .diffusion import ConditionalDenoisingUNet


class UNetFiLM(nn.Module):
    """Full U-Net model with FiLM modulation at the bottleneck."""

    def __init__(self, cfg: UNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UNetConfig()
        c = self.cfg
        # Encoder layers
        self.encoder0 = DoubleConv(c.in_channels, c.enc_channels[0])
        self.encoder1 = EncoderBlock(c.enc_channels[0], c.enc_channels[1])
        self.encoder2 = EncoderBlock(c.enc_channels[1], c.enc_channels[2])
        # Bottleneck max pooling then DoubleConv and FiLM conditioning
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(c.enc_channels[2], c.bottleneck_channels)
        self.bottleneck_film = FiLMLayer(
            robot_param_dim=c.robot_param_dim,
            feature_map_channels=c.bottleneck_channels,
        )
        # Decoder layers
        self.decoder2 = DecoderBlock(c.bottleneck_channels, c.dec_channels[0])
        self.decoder1 = DecoderBlock(c.dec_channels[0], c.dec_channels[1])
        self.decoder0 = DecoderBlock(c.dec_channels[1], c.dec_channels[2])
        # Final output convolution
        self.head = nn.Conv2d(c.dec_channels[2], c.out_channels, kernel_size=1)

    def forward(self, grid_tensor, robot_tensor):
        """Forward pass through the network.

        Parameters
        ----------
        grid_tensor : torch.Tensor
            Batched grid map tensor of shape ``[B, 4, 200, 200]``.
        robot_tensor : torch.Tensor
            Robot parameter tensor of shape ``[B, 2]``.
        """
        # Encoder path with skip connections
        x0 = self.encoder0(grid_tensor)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)

        # Bottleneck and FiLM modulation
        x3 = self.pool(x2)
        x3 = self.bottleneck(x3)
        x3 = self.bottleneck_film(x3, robot_tensor)

        # Decoder path with skip connections
        d2 = self.decoder2(x3, x2)
        d1 = self.decoder1(d2, x1)
        d0 = self.decoder0(d1, x0)

        # 1x1 convolution head producing raw logits
        logits = self.head(d0)
        return logits


class HRFiLMNet(nn.Module):
    """High-resolution network with FiLM conditioning."""

    def __init__(self, cfg: HRFiLMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HRFiLMConfig()
        c = self.cfg

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(c.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(*[ResidualBlock(32) for _ in range(4)])

        self.down2 = self._make_downsample(c.stage_channels[0], c.stage_channels[1])
        self.down3 = self._make_downsample(c.stage_channels[1], c.stage_channels[2])
        self.down4 = self._make_downsample(c.stage_channels[2], c.stage_channels[3])

        self.res2_b0 = ResidualBlock(c.stage_channels[0])
        self.res2_b1 = ResidualBlock(c.stage_channels[1])
        self.res3_b0 = ResidualBlock(c.stage_channels[0])
        self.res3_b1 = ResidualBlock(c.stage_channels[1])
        self.res3_b2 = ResidualBlock(c.stage_channels[2])
        self.res4_b0 = ResidualBlock(c.stage_channels[0])
        self.res4_b1 = ResidualBlock(c.stage_channels[1])
        self.res4_b2 = ResidualBlock(c.stage_channels[2])
        self.res4_b3 = ResidualBlock(c.stage_channels[3])

        # FiLM modules for stage3 and stage4
        self.film1 = FiLM(cond_dim=c.robot_param_dim, feat_dim=c.stage_channels[1])
        self.film2 = FiLM(cond_dim=c.robot_param_dim, feat_dim=c.stage_channels[0])

        # Context blocks after HR backbone
        self.context = nn.Sequential(
            DilatedResidualBlock(256, dilation=2),
            DilatedResidualBlock(256, dilation=4),
        )

        self.conv_fuse = nn.Conv2d(sum(c.stage_channels), 256, kernel_size=1)
        self.conv_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.PixelShuffle(2),  # 64 -> 16 channels, 200 -> 400
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, c.out_channels, kernel_size=1),
        )

    def _make_downsample(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, grid_tensor, robot_tensor):
        c = self.cfg
        # Stem and Stage1
        x = self.stem(grid_tensor)
        b0 = self.stage1(x)

        # Stage2
        b1 = self.down2(b0)
        b0 = self.res2_b0(b0)
        b1 = self.res2_b1(b1)

        # Stage3 with FiLM on branch1
        b1 = self.film1(robot_tensor, b1)
        b2 = self.down3(b1)
        b0 = self.res3_b0(b0)
        b1 = self.res3_b1(b1)
        b2 = self.res3_b2(b2)

        # Stage4 with FiLM on branch0
        b0 = self.film2(robot_tensor, b0)
        b3 = self.down4(b2)
        b0 = self.res4_b0(b0)
        b1 = self.res4_b1(b1)
        b2 = self.res4_b2(b2)
        b3 = self.res4_b3(b3)

        # Upsample all branches to highest resolution
        up1 = nn.functional.interpolate(b1, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = nn.functional.interpolate(b2, scale_factor=4, mode="bilinear", align_corners=False)
        up3 = nn.functional.interpolate(b3, scale_factor=8, mode="bilinear", align_corners=False)
        feats = torch.cat([b0, up1, up2, up3], dim=1)

        feats = self.conv_fuse(feats)
        feats = self.context(feats)
        logits = self.conv_head(feats)

        # Center crop back to 200x200
        if logits.size(-1) > 200:
            crop = (logits.size(-1) - 200) // 2
            logits = logits[:, :, crop:-crop, crop:-crop]
        return logits


def create_model(name: str, cfg_path: str | Path | None = None) -> nn.Module:
    """Factory to instantiate navigation models by name.

    Parameters
    ----------
    name : str
        Identifier of the model architecture. Supported values are
        ``"unet_film"``, ``"hr_film_net`` and ``"diffusion_unet"``.
    cfg_path : str | Path, optional
        Optional YAML file with model hyper-parameters.

    Returns
    -------
    nn.Module
        Instantiated model ready for training or inference.
    """

    # Normalize the provided model identifier to avoid issues with leading or
    # trailing whitespace or letter casing differences.
    name = name.strip().lower()
    print(name)
    if name == "unet_film":
        cfg = UNetConfig.from_yaml(cfg_path) if cfg_path else UNetConfig()
        return UNetFiLM(cfg)
    if name in {"hr_film_net", "hrfilmnet"}:
        cfg = HRFiLMConfig.from_yaml(cfg_path) if cfg_path else HRFiLMConfig()
        return HRFiLMNet(cfg)
    if name in {"diffusion_unet", "heatmap_diffusion"}:
        cfg = (
            DiffusionUNetConfig.from_yaml(cfg_path)
            if cfg_path
            else DiffusionUNetConfig()
        )
        return ConditionalDenoisingUNet(
            in_channels=cfg.in_channels,
            grid_channels=cfg.grid_channels,
            robot_param_dim=cfg.robot_param_dim,
            time_dim=cfg.time_dim,
            enc_channels=cfg.enc_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            dec_channels=cfg.dec_channels,
        )
    raise ValueError(f"Unknown model '{name}'")

