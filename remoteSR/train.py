from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from remoteSR.config import load_config
from remoteSR.data.augmentations import AugmentConfig, DegradationConfig
from remoteSR.engine import Trainer, TrainerConfig
from remoteSR.models import SemiSRConfig
from remoteSR.tasks import (
    BYOLDataConfig,
    BYOLOptimConfig,
    BYOLTaskConfig,
    DINODataConfig,
    DINOOptimConfig,
    DINOTaskConfig,
    PhiBYOLTask,
    PhiDINOTask,
    SRDataConfig,
    SRGanTask,
    SRLossConfig,
    SROptimizerConfig,
    SRTaskConfig,
)


@dataclass
class CLIArgs:
    task: str
    config: str | None
    epochs: int | None
    batch_size: int | None
    num_workers: int | None
    amp: bool | None
    grad_clip_norm: float | None
    save_dir: str | None
    save_every: int | None
    log_file: str | None
    resume: str | None
    seed: int | None
    train_lr_dir: str | None
    train_hr_dir: str | None
    phi_pretrained: str | None
    freeze_phi: bool
    lr: float | None
    weight_decay: float | None
    byol_momentum: float
    dino_tau_s: float
    dino_tau_t: float
    dino_center_momentum: float
    dino_k: int
    multi_crop: int
    crop_size: int
    local_crop_size: int
    max_translate: int
    hflip_prob: float
    vflip_prob: float
    rotate_prob: float
    blur_prob: float
    blur_kernel_min: int
    blur_kernel_max: int
    blur_sigma_min: float
    blur_sigma_max: float
    downsample_scale: int
    noise_std_min: float
    noise_std_max: float
    jitter_prob: float
    jitter_brightness: float
    jitter_contrast: float
    save_full_pretrain: bool


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_sr_task_config(args: CLIArgs) -> tuple[SRTaskConfig, TrainerConfig]:
    cfg = load_config(args.config)

    if args.train_lr_dir:
        cfg["data"]["train_lr_dir"] = args.train_lr_dir
    if args.train_hr_dir:
        cfg["data"]["train_hr_dir"] = args.train_hr_dir
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    model_cfg = SemiSRConfig(**cfg["model"])
    optim_cfg = SROptimizerConfig(**cfg["optimizer"])
    loss_cfg = SRLossConfig(**cfg["loss"])
    data_cfg = SRDataConfig(
        train_lr_dir=cfg["data"]["train_lr_dir"],
        train_hr_dir=cfg["data"]["train_hr_dir"],
        scale=cfg["data"]["scale"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        eval_enabled=cfg.get("evaluation", {}).get("enabled", False),
        eval_lr_dir=cfg.get("evaluation", {}).get("lr_dir"),
        eval_batch_size=cfg.get("evaluation", {}).get("batch_size"),
        eval_num_workers=cfg.get("evaluation", {}).get("num_workers"),
    )

    task_cfg = SRTaskConfig(
        model=model_cfg,
        optimizer=optim_cfg,
        loss=loss_cfg,
        data=data_cfg,
        phi_pretrained=args.phi_pretrained,
        freeze_phi=args.freeze_phi,
    )

    train_cfg = cfg["training"]
    trainer_cfg = TrainerConfig(
        epochs=args.epochs or train_cfg["epochs"],
        amp=args.amp if args.amp is not None else bool(train_cfg.get("amp", False)),
        grad_clip_norm=args.grad_clip_norm
        if args.grad_clip_norm is not None
        else train_cfg.get("grad_clip_norm", 1.0),
        save_every=args.save_every or train_cfg.get("save_every", 1),
        save_dir=Path(args.save_dir or train_cfg.get("output_dir", "checkpoints")),
        log_file=Path(args.log_file or train_cfg.get("log_file", "log.txt")),
        monitor="loss_total",
        mode="min",
    )

    return task_cfg, trainer_cfg


def build_pretrain_task_config(
    args: CLIArgs,
) -> tuple[BYOLTaskConfig | DINOTaskConfig, TrainerConfig]:
    if args.train_hr_dir is None:
        raise ValueError("--train_hr_dir is required for phi pretraining tasks")

    augment = AugmentConfig(
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        rotate_prob=args.rotate_prob,
        max_translate=args.max_translate,
    )

    degradation = DegradationConfig(
        blur_prob=args.blur_prob,
        blur_kernel_min=args.blur_kernel_min,
        blur_kernel_max=args.blur_kernel_max,
        blur_sigma_min=args.blur_sigma_min,
        blur_sigma_max=args.blur_sigma_max,
        downsample_scale=args.downsample_scale,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        jitter_prob=args.jitter_prob,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
    )

    num_views = max(2, args.multi_crop + 2)

    if args.task == "phi_byol":
        data_cfg = BYOLDataConfig(
            hr_dir=args.train_hr_dir,
            batch_size=args.batch_size or 16,
            num_workers=args.num_workers or 4,
            crop_size=args.crop_size,
            local_crop_size=args.local_crop_size,
            num_views=num_views,
            augment=augment,
            degradation=degradation,
        )
        opt_cfg = BYOLOptimConfig(
            lr=args.lr or 1e-3, weight_decay=args.weight_decay or 1e-4
        )
        task_cfg = BYOLTaskConfig(
            data=data_cfg,
            optimizer=opt_cfg,
            byol_momentum=args.byol_momentum,
            save_full=args.save_full_pretrain,
        )
    else:
        data_cfg = DINODataConfig(
            hr_dir=args.train_hr_dir,
            batch_size=args.batch_size or 16,
            num_workers=args.num_workers or 4,
            crop_size=args.crop_size,
            local_crop_size=args.local_crop_size,
            num_views=num_views,
            augment=augment,
            degradation=degradation,
        )
        opt_cfg = DINOOptimConfig(
            lr=args.lr or 1e-3, weight_decay=args.weight_decay or 1e-4
        )
        task_cfg = DINOTaskConfig(
            data=data_cfg,
            optimizer=opt_cfg,
            out_dim=args.dino_k,
            tau_s=args.dino_tau_s,
            tau_t=args.dino_tau_t,
            center_momentum=args.dino_center_momentum,
            ema_momentum=args.byol_momentum,
            save_full=args.save_full_pretrain,
        )

    trainer_cfg = TrainerConfig(
        epochs=args.epochs or 200,
        amp=bool(args.amp),
        grad_clip_norm=args.grad_clip_norm,
        save_every=args.save_every or 1,
        save_dir=Path(args.save_dir or "checkpoints"),
        log_file=Path(args.log_file) if args.log_file else None,
        monitor="loss",
        mode="min",
    )
    return task_cfg, trainer_cfg


def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(description="remoteSR training")
    parser.add_argument(
        "--task",
        choices=["sr_gan", "phi_byol", "phi_dino"],
        default="sr_gan",
        help="Training task",
    )
    parser.add_argument("--config", help="Path to config file (yaml/json)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--save_dir")
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--log_file")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--train_lr_dir")
    parser.add_argument("--train_hr_dir")
    parser.add_argument("--phi_pretrained")
    parser.add_argument("--freeze_phi", action="store_true")
    parser.add_argument("--no_freeze_phi", action="store_true")

    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)

    parser.add_argument("--byol_momentum", type=float, default=0.996)
    parser.add_argument("--dino_tau_s", type=float, default=0.1)
    parser.add_argument("--dino_tau_t", type=float, default=0.04)
    parser.add_argument("--dino_center_momentum", type=float, default=0.9)
    parser.add_argument("--dino_k", type=int, default=1024)
    parser.add_argument("--multi_crop", type=int, default=0)

    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--local_crop_size", type=int, default=64)
    parser.add_argument("--max_translate", type=int, default=4)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.5)
    parser.add_argument("--rotate_prob", type=float, default=0.5)

    parser.add_argument("--blur_prob", type=float, default=0.8)
    parser.add_argument("--blur_kernel_min", type=int, default=3)
    parser.add_argument("--blur_kernel_max", type=int, default=7)
    parser.add_argument("--blur_sigma_min", type=float, default=0.1)
    parser.add_argument("--blur_sigma_max", type=float, default=2.0)
    parser.add_argument("--downsample_scale", type=int, default=4)
    parser.add_argument("--noise_std_min", type=float, default=0.0)
    parser.add_argument("--noise_std_max", type=float, default=0.02)
    parser.add_argument("--jitter_prob", type=float, default=0.8)
    parser.add_argument("--jitter_brightness", type=float, default=0.1)
    parser.add_argument("--jitter_contrast", type=float, default=0.1)

    parser.add_argument("--save_full_pretrain", action="store_true")

    args = parser.parse_args()
    if not args.freeze_phi and not args.no_freeze_phi:
        args.freeze_phi = True
    else:
        args.freeze_phi = bool(args.freeze_phi) and not bool(args.no_freeze_phi)
    arg_dict = vars(args)
    arg_dict.pop("no_freeze_phi", None)
    return CLIArgs(**arg_dict)


def run_training(args: CLIArgs) -> None:
    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "sr_gan":
        task_cfg, trainer_cfg = build_sr_task_config(args)
        task = SRGanTask(task_cfg)
    elif args.task == "phi_byol":
        task_cfg, trainer_cfg = build_pretrain_task_config(args)
        task = PhiBYOLTask(task_cfg)
    else:
        task_cfg, trainer_cfg = build_pretrain_task_config(args)
        task = PhiDINOTask(task_cfg)

    trainer = Trainer(
        task=task,
        config=trainer_cfg,
        device=device,
        resume=Path(args.resume) if args.resume else None,
    )
    trainer.fit()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    run_training()
