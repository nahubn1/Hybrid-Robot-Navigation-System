from __future__ import annotations

from torch import nn

from .modules import DoubleConv, EncoderBlock, DecoderBlock, FiLMLayer


class UNetFiLM(nn.Module):
    """Full U-Net model with FiLM modulation at the bottleneck."""

    def __init__(self) -> None:
        super().__init__()
        # Encoder layers
        self.encoder0 = DoubleConv(4, 32)
        self.encoder1 = EncoderBlock(32, 64)
        self.encoder2 = EncoderBlock(64, 128)
        # Bottleneck max pooling then DoubleConv and FiLM conditioning
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(128, 256)
        self.bottleneck_film = FiLMLayer(robot_param_dim=2, feature_map_channels=256)
        # Decoder layers
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        self.decoder0 = DecoderBlock(64, 32)
        # Final output convolution
        self.head = nn.Conv2d(32, 1, kernel_size=1)

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
