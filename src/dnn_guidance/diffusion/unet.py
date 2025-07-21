from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..modules import (
    DoubleConv,
    EncoderBlock,
    DecoderBlockWithGrid,
    FiLMLayer,
)
from .time import TimestepEmbedding


class ConditionalDenoisingUNet(nn.Module):
    """U-Net backbone for conditional diffusion denoising."""

    def __init__(self, *, in_channels: int = 1, grid_channels: int = 4,
                 robot_param_dim: int = 2, time_dim: int = 256,
                 enc_channels=(32, 64, 128), bottleneck_channels: int = 256,
                 dec_channels=(128, 64, 32)) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(time_dim)
        self.time_to_ch = nn.ModuleList([
            nn.Linear(time_dim, c) for c in [*enc_channels, bottleneck_channels, *dec_channels]
        ])
        self.enc0 = DoubleConv(in_channels, enc_channels[0])
        self.enc1 = EncoderBlock(enc_channels[0], enc_channels[1])
        self.enc2 = EncoderBlock(enc_channels[1], enc_channels[2])
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(enc_channels[2], bottleneck_channels)
        self.bottle_film = FiLMLayer(robot_param_dim, bottleneck_channels)
        self.dec2 = DecoderBlockWithGrid(
            in_ch=bottleneck_channels,
            skip_ch=enc_channels[2],
            grid_ch=grid_channels,
            out_ch=dec_channels[0],
        )
        self.dec1 = DecoderBlockWithGrid(
            in_ch=dec_channels[0],
            skip_ch=enc_channels[1],
            grid_ch=grid_channels,
            out_ch=dec_channels[1],
        )
        self.dec0 = DecoderBlockWithGrid(
            in_ch=dec_channels[1],
            skip_ch=enc_channels[0],
            grid_ch=grid_channels,
            out_ch=dec_channels[2],
        )
        self.head = nn.Conv2d(dec_channels[2], 1, kernel_size=1)

    def _add_time(self, x: torch.Tensor, emb: torch.Tensor, idx: int) -> torch.Tensor:
        t = self.time_to_ch[idx](emb)[:, :, None, None]
        return x + t

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor,
                grid: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        # Encoder
        x0 = self._add_time(self.enc0(noisy), t_emb, 0)
        x1 = self._add_time(self.enc1(x0), t_emb, 1)
        x2 = self._add_time(self.enc2(x1), t_emb, 2)
        # Bottleneck
        x3 = self.pool(x2)
        x3 = self.bottleneck(x3)
        x3 = self._add_time(x3, t_emb, 3)
        x3 = self.bottle_film(x3, robot)
        # Downsample grid for skip connections
        g1 = F.interpolate(grid, size=x1.shape[-2:], mode="nearest")
        g2 = F.interpolate(grid, size=x2.shape[-2:], mode="nearest")
        g3 = F.interpolate(grid, size=noisy.shape[-2:], mode="nearest")
        # Decoder
        d2 = self.dec2(x3, x2, g2)
        d2 = self._add_time(d2, t_emb, 4)
        d1 = self.dec1(d2, x1, g1)
        d1 = self._add_time(d1, t_emb, 5)
        d0 = self.dec0(d1, x0, g3)
        d0 = self._add_time(d0, t_emb, 6)
        return self.head(d0)
