from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .generator import RCANSRGenerator

__all__ = ["CycleGANGenerator", "CycleGANDiscriminator", "CycleGANModelConfig"]


@dataclass
class CycleGANModelConfig:
    in_channels: int = 3
    base_channels: int = 64
    num_res_blocks: int = 6  # legacy fields (kept for compatibility)
    use_dropout: bool = False
    num_groups: int = 6
    blocks_per_group: int = 12
    ca_reduction: int = 16
    res_scale: float = 0.1
    scale: int = 1
    use_tanh: bool = True


def _norm_layer(num_features: int) -> nn.Module:
    return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)


class CycleGANGenerator(nn.Module):
    def __init__(self, cfg: CycleGANModelConfig) -> None:
        super().__init__()
        self.net = RCANSRGenerator(
            lr_channels=cfg.in_channels,
            hr_channels=cfg.in_channels,
            scale=cfg.scale,
            num_feats=cfg.base_channels,
            num_groups=cfg.num_groups,
            blocks_per_group=cfg.blocks_per_group,
            reduction=cfg.ca_reduction,
            res_scale=cfg.res_scale,
        )
        self.out_act = nn.Tanh() if cfg.use_tanh else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.net(x))


class CycleGANDiscriminator(nn.Module):
    def __init__(self, cfg: CycleGANModelConfig) -> None:
        super().__init__()
        c = cfg.base_channels
        layers: list[nn.Module] = [
            nn.Conv2d(cfg.in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = c
        for n in range(1, 3):
            out_ch = min(in_ch * 2, c * 8)
            stride = 1 if n == 2 else 2
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
                _norm_layer(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
