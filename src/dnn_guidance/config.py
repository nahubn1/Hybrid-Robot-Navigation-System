from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml


@dataclass
class UNetConfig:
    """Configuration for :class:`UNetFiLM`."""

    in_channels: int = 4
    enc_channels: Tuple[int, int, int] = (32, 64, 128)
    bottleneck_channels: int = 256
    robot_param_dim: int = 2
    dec_channels: Tuple[int, int, int] = (128, 64, 32)
    out_channels: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UNetConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            in_channels=int(data.get("in_channels", cls.in_channels)),
            enc_channels=tuple(data.get("enc_channels", cls.enc_channels)),
            bottleneck_channels=int(
                data.get("bottleneck_channels", cls.bottleneck_channels)
            ),
            robot_param_dim=int(data.get("robot_param_dim", cls.robot_param_dim)),
            dec_channels=tuple(data.get("dec_channels", cls.dec_channels)),
            out_channels=int(data.get("out_channels", cls.out_channels)),
        )
