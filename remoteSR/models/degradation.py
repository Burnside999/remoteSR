from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LearnablePSFDownsampler"]


class LearnablePSFDownsampler(nn.Module):
    """
    D_theta: (HR image) -> (LR observation)

    Components (a simple but effective choice):
    1) Spectral / radiometric mixing: 1x1 conv (HR channels -> LR channels)
    2) Learnable blur kernel (PSF): depthwise conv, kernel normalized by softmax (positive & sum=1)
    3) Downsample: avg_pool2d with stride=scale (x4)

    Notes:
    - This is a *model* of the sensor degradation. You train it (or jointly finetune).
    - Kernel normalization keeps it stable and interpretable.
    """
    def __init__(
        self,
        hr_channels: int,
        lr_channels: int,
        scale: int = 4,
        kernel_size: int = 9,
        share_kernel_across_channels: bool = False,
        use_radiometric_affine: bool = True,
    ):
        super().__init__()
        assert scale == 4, "Configured for x4."
        assert kernel_size % 2 == 1, "kernel_size should be odd."
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels
        self.scale = scale
        self.kernel_size = kernel_size
        self.share_kernel = share_kernel_across_channels
        self.use_radiometric_affine = use_radiometric_affine

        # Spectral mixing: HR -> LR
        self.mix = nn.Conv2d(hr_channels, lr_channels, kernel_size=1, bias=True)
        self._init_mix_identity_if_possible()

        # Learnable PSF kernel parameter (unnormalized)
        k = kernel_size
        if self.share_kernel:
            # one kernel shared across channels
            self.kernel_param = nn.Parameter(torch.zeros(1, 1, k, k))
        else:
            # per-channel kernel
            self.kernel_param = nn.Parameter(torch.zeros(lr_channels, 1, k, k))

        # Optional per-channel affine after downsample (to absorb gain/offset differences)
        if self.use_radiometric_affine:
            self.gamma = nn.Parameter(torch.ones(1, lr_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, lr_channels, 1, 1))

        self._init_kernel_centered()

    def _init_mix_identity_if_possible(self):
        # If channels match, initialize as identity mapping (helps stability).
        if self.hr_channels == self.lr_channels:
            with torch.no_grad():
                self.mix.weight.zero_()
                for c in range(self.lr_channels):
                    self.mix.weight[c, c, 0, 0] = 1.0
                if self.mix.bias is not None:
                    self.mix.bias.zero_()
        else:
            # Kaiming init
            nn.init.kaiming_normal_(self.mix.weight, a=0.0, mode="fan_in", nonlinearity="linear")
            if self.mix.bias is not None:
                nn.init.zeros_(self.mix.bias)

    def _init_kernel_centered(self):
        # Initialize blur kernel close to delta (center=1), then softmax -> near-identity.
        with torch.no_grad():
            self.kernel_param.zero_()
            k = self.kernel_size
            cy = k // 2
            cx = k // 2
            self.kernel_param[..., cy, cx] = 5.0  # a bit peaky so softmax concentrates near center

    def normalized_kernel(self) -> torch.Tensor:
        """
        Returns:
            kernel: (lr_channels, 1, k, k) if not shared,
                    otherwise expanded to (lr_channels, 1, k, k)
        """
        k = self.kernel_size
        if self.share_kernel:
            # (1,1,k,k) -> (lr_channels,1,k,k)
            raw = self.kernel_param.view(1, -1)  # (1, k*k)
            w = F.softmax(raw, dim=1).view(1, 1, k, k)
            w = w.expand(self.lr_channels, 1, k, k).contiguous()
            return w
        else:
            raw = self.kernel_param.view(self.lr_channels, -1)  # (C, k*k)
            w = F.softmax(raw, dim=1).view(self.lr_channels, 1, k, k)
            return w

    def forward(self, x_hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hr: (B, hr_channels, H_hr, W_hr)

        Returns:
            y_lr_hat: (B, lr_channels, H_hr/4, W_hr/4)
        """
        # 1) spectral mixing
        z = self.mix(x_hr)  # (B, lr_channels, H, W)

        # 2) blur with depthwise conv
        kernel = self.normalized_kernel()  # (lr_channels,1,k,k)
        pad = self.kernel_size // 2
        z = F.pad(z, (pad, pad, pad, pad), mode="reflect")
        z = F.conv2d(z, kernel, bias=None, stride=1, padding=0, groups=self.lr_channels)

        # 3) downsample (avg pooling)
        y = F.avg_pool2d(z, kernel_size=self.scale, stride=self.scale)

        # 4) optional affine
        if self.use_radiometric_affine:
            y = y * self.gamma + self.beta
        return y