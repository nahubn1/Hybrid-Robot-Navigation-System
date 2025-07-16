"""Reusable neural network building blocks for U-Net-like architectures."""

from __future__ import annotations

import torch
from torch import nn


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
