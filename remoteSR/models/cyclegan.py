from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["CycleGANGenerator", "CycleGANDiscriminator", "CycleGANModelConfig"]


@dataclass
class CycleGANModelConfig:
    in_channels: int = 3
    base_channels: int = 64
    num_res_blocks: int = 6
    use_dropout: bool = False


def _norm_layer(num_features: int) -> nn.Module:
    return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int, use_dropout: bool) -> None:
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=True),
            _norm_layer(channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=True),
            _norm_layer(channels),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CycleGANGenerator(nn.Module):
    def __init__(self, cfg: CycleGANModelConfig) -> None:
        super().__init__()
        c = cfg.base_channels
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(cfg.in_channels, c, kernel_size=7, bias=True),
            _norm_layer(c),
            nn.ReLU(inplace=True),
        ]
        in_ch = c
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
                _norm_layer(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        for _ in range(cfg.num_res_blocks):
            layers.append(ResnetBlock(in_ch, cfg.use_dropout))

        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                _norm_layer(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, cfg.in_channels, kernel_size=7, bias=True),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
