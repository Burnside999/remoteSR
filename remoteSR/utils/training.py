from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .io import save_tensor_as_png


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


@torch.no_grad()
def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = (
            value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        )
    return out


def train_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """
    One forward/backward step for semi-supervised SR.
    """
    model.train()
    batch = _move_to_device(batch, device)

    y_lr = batch["y_lr"]
    x_ls_hr = batch["x_ls_hr"]
    mask_lr = batch.get("mask_lr", None)
    mask_hr = batch.get("mask_hr", None)
    x_hr_gt = batch.get("x_hr_gt", None)
    has_hr_gt = batch.get("has_hr_gt", None)

    optimizer.zero_grad(set_to_none=True)

    if scaler is None:
        total, logs = criterion(
            y_lr,
            x_ls_hr,
            mask_lr=mask_lr,
            mask_hr=mask_hr,
            x_hr_gt=x_hr_gt,
            has_hr_gt=has_hr_gt,
        )
        total.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    else:
        with torch.amp.autocast("cuda", enabled=amp):
            total, logs = criterion(
                y_lr,
                x_ls_hr,
                mask_lr=mask_lr,
                mask_hr=mask_hr,
                x_hr_gt=x_hr_gt,
                has_hr_gt=has_hr_gt,
            )

        scaler.scale(total).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

    return {k: float(v.item()) for k, v in logs.items()}


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """
    Minimal epoch loop for a given dataloader.
    """
    running: dict[str, float] = {}
    count = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        logs = train_step(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            batch=batch,
            device=device,
            scaler=scaler,
            amp=amp,
            grad_clip_norm=grad_clip_norm,
        )
        count += 1
        for key, value in logs.items():
            running[key] = running.get(key, 0.0) + value

    for key in running:
        running[key] /= max(1, count)
    return running


@torch.no_grad()
def evaluate_lr_reconstruction(
    model: nn.Module,
    dataloader,
    device: torch.device,
    desc: str = "Eval",
) -> dict[str, float]:
    """
    Evaluate measurement consistency on LR-only data by comparing the model's LR reconstruction
    with the input LR frames. This avoids direct HR comparisons when LR/HR are misaligned.
    """
    was_training = model.training
    model.eval()

    output = "testres"

    os.makedirs(output, exist_ok=True)

    batch_i = 0

    for batch in tqdm(dataloader, desc=desc):
        batch = _move_to_device(batch, device)
        y_lr = batch["y_lr"]
        x_hat = model(y_lr)
        save_tensor_as_png(x_hat, os.path.join(output, f"{batch_i}.png"))
        batch_i += 1

    if was_training:
        model.train()
