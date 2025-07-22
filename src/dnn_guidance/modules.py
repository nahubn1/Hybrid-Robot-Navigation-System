"""Reusable neural network building blocks for U-Net-like architectures."""

from __future__ import annotations

import torch
from torch import nn


class SELayer(nn.Module):
    """Squeeze-and-Excitation module."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder stage consisting of max pooling followed by :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder stage with upsampling, concatenation and :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + out_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x1 : torch.Tensor
            Feature map from the previous decoder layer.
        x2 : torch.Tensor
            Feature map from the corresponding encoder layer (skip connection).
        """
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    This layer modulates a feature map using a vector of robot parameters. It
    computes ``gamma`` and ``beta`` through a small MLP and applies the FiLM
    transformation: ``x * (1 + gamma) + beta``.

    Parameters
    ----------
    robot_param_dim : int
        Dimensionality of the input robot parameter vector.
    feature_map_channels : int
        Number of channels in the feature map to be modulated.
    """

    def __init__(self, robot_param_dim: int, feature_map_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(robot_param_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_map_channels * 2),
        )

    def forward(
        self, feature_map: torch.Tensor, robot_vector: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM modulation to ``feature_map`` using ``robot_vector``."""

        params = self.mlp(robot_vector)
        gamma, beta = torch.chunk(params, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1 + gamma) + beta


class FiLM(nn.Module):
    """FiLM conditioning module used in HR-FiLM-Net."""

    def __init__(self, cond_dim: int = 2, feat_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feat_dim * 2),
        )

    def forward(self, cond: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return feat * (1 + gamma) + beta


class ResidualBlock(nn.Module):
    """Simple residual block with optional SE and dropout."""

    def __init__(self, channels: int, *, dilation: int = 1, se: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SELayer(channels) if se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.dropout(out)
        out += x
        return self.relu(out)


class DilatedResidualBlock(ResidualBlock):
    """Residual block with configurable dilation."""

    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__(channels, dilation=dilation)


class DecoderBlockWithGrid(nn.Module):
    """Decoder block that concatenates skip and grid features."""

    def __init__(self, in_ch: int, skip_ch: int, grid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch + grid_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip, grid], dim=1)
        return self.conv(x)
