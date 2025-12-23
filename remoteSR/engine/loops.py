from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from tqdm import tqdm

from remoteSR.utils.io import save_tensor_as_png


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
    trace_step: int | None = None,
    trace_dir: Path | None = None,
) -> dict[str, float]:
    """
    Minimal epoch loop for a given dataloader.
    """
    running: dict[str, float] = {}
    count = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        step_index = count + 1
        if trace_step is not None and step_index == trace_step:
            trace_dir = trace_dir or Path("traces")
            trace_dir.mkdir(parents=True, exist_ok=True)
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
            ) as prof:
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
                prof.step()
        else:
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

    output_dir = Path("testres")
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_i, batch in enumerate(tqdm(dataloader, desc=desc)):
        batch = _move_to_device(batch, device)
        y_lr = batch["y_lr"]
        x_hat = model(y_lr)
        save_tensor_as_png(x_hat, str(output_dir / f"{batch_i}.png"))

    if was_training:
        model.train()
