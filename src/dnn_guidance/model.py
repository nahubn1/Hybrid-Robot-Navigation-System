from __future__ import annotations

from torch import nn

from .config import UNetConfig
from .modules import DoubleConv, EncoderBlock, DecoderBlock, FiLMLayer


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

