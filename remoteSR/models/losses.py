from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from remoteSR.utils.io import save_tensor_as_png, stat

__all__ = ["LossWeights", "SemiSRLoss"]


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/target: (B, C, H, W)
    mask: (B, 1, H, W) or (B, C, H, W) or None
    """
    diff = (pred - target).abs()
    if mask is None:
        return diff.mean()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and diff.shape[1] != 1:
        mask = mask.expand(-1, diff.shape[1], -1, -1)

    diff = diff * mask
    denom = mask.sum() * (1.0 if mask.shape[1] == diff.shape[1] else diff.shape[1])
    return diff.sum() / (denom + eps)


def total_variation_loss(x: torch.Tensor, mask: Optional[torch.Tensor] = None, reduction: str = "mean") -> torch.Tensor:
    """
    Total variation loss on an image tensor.
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mh = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mw = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        if mh.shape[1] == 1 and dh.shape[1] != 1:
            mh = mh.expand(-1, dh.shape[1], -1, -1)
        if mw.shape[1] == 1 and dw.shape[1] != 1:
            mw = mw.expand(-1, dw.shape[1], -1, -1)
        dh = dh * mh
        dw = dw * mw
        if reduction == "mean":
            return (dh.sum() / (mh.sum() + 1e-8) + dw.sum() / (mw.sum() + 1e-8)) * 0.5
        return dh.sum() + dw.sum()

    if reduction == "mean":
        return dh.mean() + dw.mean()
    return dh.sum() + dw.sum()


def sobel(x: torch.Tensor, mode: str = "mag", eps: float = 1e-6) -> torch.Tensor:
    """
    Depthwise Sobel gradient.
    mode: "x", "y", "mag" (|gx|+|gy|), or "sqrt" (sqrt(gx^2+gy^2)).
    """
    assert x.dim() == 4, f"Expect (B,C,H,W), got {x.shape}"
    b, c, h, w = x.shape
    dtype, device = x.dtype, x.device

    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)

    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)

    if mode == "x":
        return gx
    if mode == "y":
        return gy
    if mode == "sqrt":
        return torch.sqrt(gx * gx + gy * gy + eps)
    return gx.abs() + gy.abs()


class ContextualLoss(nn.Module):
    """
    Contextual Loss for non-aligned feature matching with optional random sampling.
    """
    def __init__(self, bandwidth: float = 0.1, max_samples: int = 1024, eps: float = 1e-5):
        super().__init__()
        self.bandwidth = float(bandwidth)
        self.max_samples = int(max_samples)
        self.eps = float(eps)

    @torch.no_grad()
    def _rand_idx(self, n: int, k: int, device: torch.device) -> torch.Tensor:
        if k >= n:
            return torch.arange(n, device=device)
        return torch.randperm(n, device=device)[:k]

    def forward(self, y_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        assert y_feat.shape == x_feat.shape, f"{y_feat.shape} != {x_feat.shape}"
        b, c, h, w = y_feat.shape
        n = h * w
        k = min(self.max_samples, n)

        y_feat = y_feat.float()
        x_feat = x_feat.float()

        y = y_feat.view(b, c, n)
        x = x_feat.view(b, c, n)

        if k < n:
            idx_y = self._rand_idx(n, k, y.device)
            idx_x = self._rand_idx(n, k, y.device)
            y = y[:, :, idx_y]
            x = x[:, :, idx_x]

        y = F.normalize(y, dim=1, eps=self.eps)
        x = F.normalize(x, dim=1, eps=self.eps)

        dist = 1.0 - torch.bmm(y.transpose(1, 2), x)
        dist = dist.clamp(min=0.0, max=2.0)

        dist_min = dist.min(dim=2, keepdim=True).values.clamp(min=self.eps)
        dist_tilde = dist / dist_min

        log_w = (1.0 - dist_tilde) / max(self.bandwidth, 1e-4)
        log_w = log_w.clamp(min=-50.0, max=50.0)

        w = torch.exp(log_w)
        cx = w / (w.sum(dim=2, keepdim=True) + self.eps)

        cx_i = cx.max(dim=2).values
        cx_mean = cx_i.mean(dim=1)
        loss = (-torch.log(cx_mean + self.eps)).mean()
        return loss


@dataclass
class LossWeights:
    lambda_lr: float = 1.0
    lambda_match: float = 0.1
    lambda_ctx: float = 0.1
    lambda_tv: float = 1e-6
    lambda_pix: float = 0.0
    lambda_grad: float = 0.0


class SemiSRLoss(nn.Module):
    """
    Combined loss for semi-supervised SR.
    """
    def __init__(
        self,
        model: nn.Module,
        weights: LossWeights,
        cx_bandwidth: float = 0.1,
        cx_max_samples: int = 1024,
        use_mask: bool = False,
    ):
        super().__init__()
        self.model = model
        self.w = weights
        self.use_mask = bool(use_mask)
        self.cx = ContextualLoss(bandwidth=cx_bandwidth, max_samples=cx_max_samples)

    def forward(
        self,
        y_lr: torch.Tensor,
        x_ls_hr: torch.Tensor,
        mask_lr: Optional[torch.Tensor] = None,
        mask_hr: Optional[torch.Tensor] = None,
        x_hr_gt: Optional[torch.Tensor] = None,
        has_hr_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_hat, y_recon = self.model(y_lr, return_lr_recon=True, return_features=False)

        if torch.rand(()) < 0.01:
            save_tensor_as_png(x_hat, "debug_out/x_hat_step.png")
            save_tensor_as_png(torch.nn.functional.interpolate(y_lr, scale_factor=4, mode="nearest"), "debug_out/y_lr_up.png")
            save_tensor_as_png(x_ls_hr, "debug_out/x_ls_hr.png")

        l_lr = masked_l1(y_recon, y_lr, mask_lr) if (self.use_mask and mask_lr is not None) else F.l1_loss(y_recon, y_lr)

        f_hat = self.model.phi(x_hat)
        f_ls = self.model.phi(x_ls_hr).detach()

        q = F.normalize(f_hat, dim=1, eps=1e-6)
        k = F.normalize(f_ls, dim=1, eps=1e-6)

        pool = 2
        if pool > 1:
            q = F.avg_pool2d(q, pool, pool)
            k = F.avg_pool2d(k, pool, pool)

        k_aligned = self.model.aligner(q, k)
        l_match = F.l1_loss(q.float(), k_aligned)

        l_ctx = self.cx(q.float(), k.float())

        if self.use_mask and mask_hr is not None:
            l_tv = total_variation_loss(x_hat.float(), mask_hr)
        else:
            l_tv = total_variation_loss(x_hat.float())

        l_pix = x_hat.new_tensor(0.0)
        if (x_hr_gt is not None) and (has_hr_gt is not None) and (self.w.lambda_pix > 0):
            has = has_hr_gt.view(-1).float() if has_hr_gt.dim() > 1 else has_hr_gt.float()
            per = (x_hat - x_hr_gt).abs().mean(dim=(1, 2, 3))
            denom = has.sum().clamp(min=1.0)
            l_pix = (per * has).sum() / denom

        xh = F.avg_pool2d(x_hat, 2, 2)
        xt = F.avg_pool2d(x_ls_hr.detach(), 2, 2)
        gxh = sobel(xh, "mag")
        gxt = sobel(xt, "mag")
        l_grad = F.l1_loss(gxh, gxt)

        total = (
            self.w.lambda_lr * l_lr
            + self.w.lambda_match * l_match
            + self.w.lambda_ctx * l_ctx
            + self.w.lambda_tv * l_tv
            + self.w.lambda_pix * l_pix
            + self.w.lambda_grad * l_grad
        )

        logs = {
            "loss_total": total.detach(),
            "loss_lr": l_lr.detach(),
            "loss_match": l_match.detach(),
            "loss_ctx": l_ctx.detach(),
            "loss_tv": l_tv.detach(),
            "loss_pix": l_pix.detach(),
            "loss_grad": l_grad.detach(),
        }
        return total, logs
