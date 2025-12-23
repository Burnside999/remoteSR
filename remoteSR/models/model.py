from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .align import SoftShiftAligner
from .degradation import LearnablePSFDownsampler
from .feature import TinySwinPhi
from .generator import RCANSRGenerator

__all__ = ["SemiSRConfig", "SemiSupervisedSRModel"]


@dataclass
class SemiSRConfig:
    lr_channels: int
    hr_channels: int
    scale: int = 4

    # Generator
    num_feats: int = 96
    num_groups: int = 6
    blocks_per_group: int = 12
    ca_reduction: int = 16
    res_scale: float = 0.1

    # Degradation
    kernel_size: int = 9
    share_kernel: bool = False
    use_radiometric_affine: bool = True

    # Distillation feature & alignment
    distill_feat_channels: int = 64
    distill_feat_layers: int = 3
    distill_feat_downsample: int = 2  # set 2 to save memory
    align_window: int = 9
    align_temperature: float = 0.07
    align_normalize: bool = True


class SemiSupervisedSRModel(nn.Module):
    """
    The model container for your scheme A.

    - G: SR generator (Himawari LR -> HR)
    - D: degradation model (HR -> Himawari-like LR)
    - phi: feature extractor for distillation
    - aligner: local soft matching aligner

    In training you typically do:
      x_hat = model.G(y)
      y_recon = model.D(x_hat)
      L_lr = |y_recon - y|
      f_hat = model.phi(x_hat)
      f_ls  = model.phi(x_landsat)
      L_match = |f_hat - align(f_hat, f_ls_detach)|   (teacher stopgrad)
    """
    def __init__(self, cfg: SemiSRConfig):
        super().__init__()
        assert cfg.scale == 4, "This wrapper is configured for x4."

        self.cfg = cfg
        print(self.cfg)
        self.G = RCANSRGenerator(
            lr_channels=cfg.lr_channels,
            hr_channels=cfg.hr_channels,
            scale=cfg.scale,
            num_feats=cfg.num_feats,
            num_groups=cfg.num_groups,
            blocks_per_group=cfg.blocks_per_group,
            reduction=cfg.ca_reduction,
            res_scale=cfg.res_scale,
        )
        self.D = LearnablePSFDownsampler(
            hr_channels=cfg.hr_channels,
            lr_channels=cfg.lr_channels,
            scale=cfg.scale,
            kernel_size=cfg.kernel_size,
            share_kernel_across_channels=cfg.share_kernel,
            use_radiometric_affine=cfg.use_radiometric_affine,
        )
        self.phi = TinySwinPhi(
            in_channels=cfg.hr_channels,
            embed_dim=64,
            depth=4,
            num_heads=4,
            window_size=8,
            mlp_ratio=2.0,
        )
        
        self.aligner = SoftShiftAligner(
            window_size=cfg.align_window,
            temperature=cfg.align_temperature,
            normalize=cfg.align_normalize,
        )
        '''
        self.aligner = SoftShiftAlignerStreaming(
            window_size=cfg.align_window,
            temperature=cfg.align_temperature
        )
        '''

    def forward(
        self,
        y_lr: torch.Tensor,
        return_lr_recon: bool = False,
        return_features: bool = False,
    ):
        """
        Forward for inference or to get reconstruction for measurement consistency.
        """
        if return_features:
            x_hat, feats = self.G(y_lr, return_features=True)
        else:
            x_hat = self.G(y_lr, return_features=False)
            feats = None

        if not return_lr_recon:
            return (x_hat, feats) if return_features else x_hat

        y_recon = self.D(x_hat)
        if return_features:
            return x_hat, y_recon, feats
        return x_hat, y_recon
