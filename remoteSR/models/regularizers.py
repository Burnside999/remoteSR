from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["masked_l1", "sobel", "total_variation_loss", "texture_penalty"]


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
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


def total_variation_loss(
    x: torch.Tensor, mask: torch.Tensor | None = None, reduction: str = "mean"
) -> torch.Tensor:
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

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

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


def _local_variance(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    var = F.avg_pool2d((x - mean) ** 2, kernel_size, stride=1, padding=pad)
    return var


def _standardize_per_image(x, eps=1e-6):
    # x: (B,C,H,W)
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + eps)


def texture_penalty(
    x_hat: torch.Tensor, x_ls_hr: torch.Tensor, kernel_size: int = 9
) -> torch.Tensor:
    x_hat_n = _standardize_per_image(x_hat.float())
    x_ls_n = _standardize_per_image(x_ls_hr.float())

    # 局部方差（texture 强弱）
    var_hat = _local_variance(x_hat_n, kernel_size=kernel_size)
    var_ls = _local_variance(x_ls_n, kernel_size=kernel_size)

    # 更稳定（可选）
    var_hat = torch.log1p(var_hat)
    var_ls = torch.log1p(var_ls)

    var_hat_q = F.normalize(var_hat, dim=1, eps=1e-6)
    var_ls_k = F.normalize(var_ls, dim=1, eps=1e-6)

    l_texture = F.l1_loss(var_hat_q, var_ls_k)

    return l_texture
