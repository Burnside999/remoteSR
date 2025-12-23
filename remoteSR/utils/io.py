from __future__ import annotations

import os

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
