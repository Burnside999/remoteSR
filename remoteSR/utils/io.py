from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import torch


@torch.no_grad()
def save_tensor_as_png(x: torch.Tensor, path: str) -> None:
    """
    Save a tensor as an image. Accepts (C,H,W) or (1,C,H,W) tensors in [0,1].
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().cpu()

    img = x[:3] if x.shape[0] >= 3 else x.repeat(3, 1, 1)
    img = img.clamp(0, 1)
    img = (img * 255.0).byte()

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    from torchvision.utils import save_image

    save_image(img.float() / 255.0, path)


@torch.no_grad()
def stat(name: str, x: torch.Tensor, n: int = 5) -> None:
    """
    Quick tensor statistics for debugging.
    """
    x_f = x.detach().float()
    finite = torch.isfinite(x_f)
    msg = (
        f"{name:12s} "
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={x_f[finite].min().item():.4f} "
        f"max={x_f[finite].max().item():.4f} "
        f"mean={x_f[finite].mean().item():.4f} "
        f"std={x_f[finite].std().item():.4f} "
        f"finite={finite.float().mean().item() * 100:.2f}%"
    )
    print(msg)
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_image_tensor(
    path: Path,
    resize_factor: float,
    device: torch.device,
    percentile_clip: Iterable[float] | None = None,
) -> torch.Tensor | None:
    """
    Load an image from disk, optionally resize it and clip percentile ranges, and
    return a float32 tensor on the requested device.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_factor and resize_factor != 1.0:
        new_h = max(1, int(img.shape[0] * resize_factor))
        new_w = max(1, int(img.shape[1] * resize_factor))
        img = cv2.resize(img, (new_w, new_h))
    img = img.astype(np.float32) / 255.0

    if percentile_clip is not None:
        lo, hi = percentile_clip
        lo_p, hi_p = (float(lo), float(hi))
        lo_v = np.percentile(img, lo_p)
        hi_v = np.percentile(img, hi_p)
        img = (img - lo_v) / (hi_v - lo_v + 1e-8)
        img = np.clip(img, 0, 1)

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor
