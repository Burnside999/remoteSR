from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr_g: float = 2e-4,
    lr_d: float = 1e-4,
    lr_phi: float = 2e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Build an optimizer with separate parameter groups for G, D, and phi (if present).
    """
    params = []
    if hasattr(model, "G"):
        params.append({"params": model.G.parameters(), "lr": lr_g})
    else:
        params.append({"params": model.parameters(), "lr": lr_g})

    if hasattr(model, "D"):
        params.append({"params": model.D.parameters(), "lr": lr_d})

    if hasattr(model, "phi"):
        params.append({"params": model.phi.parameters(), "lr": lr_phi})

    return torch.optim.AdamW(params, betas=(0.9, 0.99), weight_decay=weight_decay)
