from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RCANSRGenerator"]


def _default_act():
    return nn.ReLU(inplace=True)


class ChannelAttention(nn.Module):
    """RCAN-style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(self.avg_pool(x))
        return x * w


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(
        self, channels: int, reduction: int = 16, act: nn.Module | None = None
    ):
        super().__init__()
        act = act if act is not None else _default_act()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.act = act
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.ca = ChannelAttention(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.act(self.conv1(x)))
        res = self.ca(res)
        return x + res


class ResizeConvUpsampler(nn.Module):
    def __init__(self, channels: int, scale: int = 4, act=None):
        super().__init__()
        assert scale in (2, 4, 8)
        act = act if act is not None else nn.ReLU(inplace=True)

        layers = []
        s = scale
        c = channels
        while s > 1:
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c, c, 3, padding=1, bias=True),
                act,
            ]
            s //= 2
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualGroup(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        reduction: int = 16,
        act=None,
        res_scale: float = 1.0,
    ):
        super().__init__()
        act = act if act is not None else nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(
            *[RCAB(channels, reduction=reduction, act=act) for _ in range(num_blocks)]
        )
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.res_scale = float(res_scale)

    def forward(self, x):
        res = self.conv(self.blocks(x))
        return x + res * self.res_scale


class RCANSRGenerator(nn.Module):
    """
    A stable CNN SR backbone for remote sensing (often easier than Transformers).
    - Head conv
    - N RCAB blocks
    - Global residual
    - ResizeConv upsampler (x4)
    - Tail conv

    You can optionally return intermediate features for distillation.
    """

    def __init__(
        self,
        lr_channels: int,
        hr_channels: int,
        scale: int = 4,
        num_feats: int = 64,
        num_groups: int = 5,  # NEW
        blocks_per_group: int = 10,  # NEW
        reduction: int = 16,
        res_scale: float = 0.1,  # NEW
        act: nn.Module | None = None,
    ):
        super().__init__()
        assert scale == 4, "This implementation is configured for x4 as you requested."
        act = act if act is not None else _default_act()

        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.scale = scale

        self.head = nn.Conv2d(lr_channels, num_feats, 3, padding=1, bias=True)

        self.body = nn.Sequential(
            *[
                ResidualGroup(
                    num_feats,
                    blocks_per_group,
                    reduction=reduction,
                    act=act,
                    res_scale=res_scale,
                )
                for _ in range(num_groups)
            ]
        )
        self.body_conv = nn.Conv2d(num_feats, num_feats, 3, padding=1, bias=True)

        self.upsampler = ResizeConvUpsampler(num_feats, scale=scale, act=act)
        self.tail = nn.Conv2d(num_feats, hr_channels, 3, padding=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init is usually safe for SR CNN
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        y_lr: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            y_lr: (B, lr_channels, H, W)
            return_features: if True, also return a dict with features:
                - feat_lr: feature map at LR resolution (before upsample)
                - feat_hr: feature map at HR resolution (after upsample, before tail)

        Returns:
            x_hat_hr: (B, hr_channels, H*4, W*4)
        """
        feat0 = self.head(y_lr)  # (B, F, H, W)
        body = self.body(feat0)  # (B, F, H, W)
        feat_lr = self.body_conv(body) + feat0  # global residual
        feat_hr = self.upsampler(feat_lr)  # (B, F, 4H, 4W)
        x_hat = self.tail(feat_hr)  # (B, C_hr, 4H, 4W)

        # x_hat = torch.sigmoid(x_hat)  # 强制到 [0,1]

        if not return_features:
            return x_hat

        feats = {
            "feat_lr": feat_lr,
            "feat_hr": feat_hr,
        }
        return x_hat, feats
