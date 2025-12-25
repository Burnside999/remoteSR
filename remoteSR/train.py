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
    phi_checkpoint: str | None
    g_checkpoint: str | None
    d_checkpoint: str | None
    freeze_phi_epochs: int | None
    freeze_g_epochs: int | None
    freeze_d_epochs: int | None
    lr: float | None
    weight_decay: float | None
    byol_momentum: float | None
    dino_tau_s: float | None
    dino_tau_t: float | None
    dino_center_momentum: float | None
    dino_k: int | None
    multi_crop: int | None
    crop_size: int | None
    local_crop_size: int | None
    max_translate: int | None
    hflip_prob: float | None
    vflip_prob: float | None
    rotate_prob: float | None
    blur_prob: float | None
    blur_kernel_min: int | None
    blur_kernel_max: int | None
    blur_sigma_min: float | None
    blur_sigma_max: float | None
    downsample_prob: float | None
    downsample_scale: int | None
    noise_std_min: float | None
    noise_std_max: float | None
    jitter_prob: float | None
    jitter_brightness: float | None
    jitter_contrast: float | None
    save_full_pretrain: bool | None


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

    train_cfg = cfg["training"]
    checkpoints_cfg = train_cfg.get("checkpoints", {})
    freeze_cfg = train_cfg.get("freeze_epochs", {})

    freeze_phi_epochs = (
        args.freeze_phi_epochs
        if args.freeze_phi_epochs is not None
        else freeze_cfg.get("phi", 0)
    )
    freeze_g_epochs = (
        args.freeze_g_epochs
        if args.freeze_g_epochs is not None
        else freeze_cfg.get("g", 0)
    )
    freeze_d_epochs = (
        args.freeze_d_epochs
        if args.freeze_d_epochs is not None
        else freeze_cfg.get("d", 0)
    )

    if args.freeze_phi and args.freeze_phi_epochs is None:
        freeze_phi_epochs = -1
    if args.freeze_phi_epochs == -1 and args.no_freeze_phi:
        freeze_phi_epochs = 0
    if args.no_freeze_phi and args.freeze_phi_epochs is None:
        freeze_phi_epochs = 0

    phi_checkpoint = args.phi_checkpoint or checkpoints_cfg.get("phi")
    g_checkpoint = args.g_checkpoint or checkpoints_cfg.get("g")
    d_checkpoint = args.d_checkpoint or checkpoints_cfg.get("d")

    task_cfg = SRTaskConfig(
        model=model_cfg,
        optimizer=optim_cfg,
        loss=loss_cfg,
        data=data_cfg,
        phi_pretrained=args.phi_pretrained,
        checkpoint_phi=phi_checkpoint,
        checkpoint_g=g_checkpoint,
        checkpoint_d=d_checkpoint,
        freeze_phi_epochs=int(freeze_phi_epochs),
        freeze_g_epochs=int(freeze_g_epochs),
        freeze_d_epochs=int(freeze_d_epochs),
    )

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
    cfg = load_config(args.config)
    pretrain_cfg = cfg.get("pretrain", {})
    pretrain_data = pretrain_cfg.get("data", {})

    train_hr_dir = args.train_hr_dir or pretrain_data.get("hr_dir")
    if train_hr_dir is None:
        raise ValueError(
            "--train_hr_dir is required for phi pretraining tasks (or set pretrain.data.hr_dir)"
        )

    augment = AugmentConfig(
        hflip_prob=args.hflip_prob
        if args.hflip_prob is not None
        else pretrain_cfg.get("augment", {}).get("hflip_prob", 0.5),
        vflip_prob=args.vflip_prob
        if args.vflip_prob is not None
        else pretrain_cfg.get("augment", {}).get("vflip_prob", 0.5),
        rotate_prob=args.rotate_prob
        if args.rotate_prob is not None
        else pretrain_cfg.get("augment", {}).get("rotate_prob", 0.5),
        max_translate=args.max_translate
        if args.max_translate is not None
        else pretrain_cfg.get("augment", {}).get("max_translate", 4),
    )

    degradation = DegradationConfig(
        blur_prob=args.blur_prob
        if args.blur_prob is not None
        else pretrain_cfg.get("degradation", {}).get("blur_prob", 0.8),
        blur_kernel_min=args.blur_kernel_min
        if args.blur_kernel_min is not None
        else pretrain_cfg.get("degradation", {}).get("blur_kernel_min", 3),
        blur_kernel_max=args.blur_kernel_max
        if args.blur_kernel_max is not None
        else pretrain_cfg.get("degradation", {}).get("blur_kernel_max", 7),
        blur_sigma_min=args.blur_sigma_min
        if args.blur_sigma_min is not None
        else pretrain_cfg.get("degradation", {}).get("blur_sigma_min", 0.1),
        blur_sigma_max=args.blur_sigma_max
        if args.blur_sigma_max is not None
        else pretrain_cfg.get("degradation", {}).get("blur_sigma_max", 2.0),
        downsample_prob=args.downsample_prob
        if args.downsample_prob is not None
        else pretrain_cfg.get("degradation", {}).get("downsample_prob", 0.0),
        downsample_scale=args.downsample_scale
        if args.downsample_scale is not None
        else pretrain_cfg.get("degradation", {}).get("downsample_scale", 4),
        noise_std_min=args.noise_std_min
        if args.noise_std_min is not None
        else pretrain_cfg.get("degradation", {}).get("noise_std_min", 0.0),
        noise_std_max=args.noise_std_max
        if args.noise_std_max is not None
        else pretrain_cfg.get("degradation", {}).get("noise_std_max", 0.02),
        jitter_prob=args.jitter_prob
        if args.jitter_prob is not None
        else pretrain_cfg.get("degradation", {}).get("jitter_prob", 0.8),
        jitter_brightness=args.jitter_brightness
        if args.jitter_brightness is not None
        else pretrain_cfg.get("degradation", {}).get("jitter_brightness", 0.1),
        jitter_contrast=args.jitter_contrast
        if args.jitter_contrast is not None
        else pretrain_cfg.get("degradation", {}).get("jitter_contrast", 0.1),
    )

    multi_crop = (
        args.multi_crop
        if args.multi_crop is not None
        else pretrain_data.get("multi_crop", 0)
    )
    num_views = max(2, multi_crop + 2)

    crop_size = (
        args.crop_size
        if args.crop_size is not None
        else pretrain_data.get("crop_size", 96)
    )
    local_crop_size = (
        args.local_crop_size
        if args.local_crop_size is not None
        else pretrain_data.get("local_crop_size", 64)
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else pretrain_data.get("batch_size", 16)
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else pretrain_data.get("num_workers", 4)
    )

    optim_cfg = pretrain_cfg.get("optimizer", {})
    lr = args.lr if args.lr is not None else optim_cfg.get("lr", 1e-3)
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else optim_cfg.get("weight_decay", 1e-4)
    )

    if args.task == "phi_byol":
        data_cfg = BYOLDataConfig(
            hr_dir=train_hr_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            crop_size=crop_size,
            local_crop_size=local_crop_size,
            num_views=num_views,
            augment=augment,
            degradation=degradation,
        )
        opt_cfg = BYOLOptimConfig(lr=lr, weight_decay=weight_decay)
        byol_cfg = pretrain_cfg.get("byol", {})
        byol_momentum = (
            args.byol_momentum
            if args.byol_momentum is not None
            else byol_cfg.get("momentum", 0.996)
        )
        task_cfg = BYOLTaskConfig(
            data=data_cfg,
            optimizer=opt_cfg,
            byol_momentum=byol_momentum,
            save_full=args.save_full_pretrain
            if args.save_full_pretrain is not None
            else bool(pretrain_cfg.get("save_full_pretrain", False)),
        )
    else:
        data_cfg = DINODataConfig(
            hr_dir=train_hr_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            crop_size=crop_size,
            local_crop_size=local_crop_size,
            num_views=num_views,
            augment=augment,
            degradation=degradation,
        )
        opt_cfg = DINOOptimConfig(lr=lr, weight_decay=weight_decay)
        dino_cfg = pretrain_cfg.get("dino", {})
        tau_s = (
            args.dino_tau_s
            if args.dino_tau_s is not None
            else dino_cfg.get("tau_s", 0.1)
        )
        tau_t = (
            args.dino_tau_t
            if args.dino_tau_t is not None
            else dino_cfg.get("tau_t", 0.04)
        )
        center_momentum = (
            args.dino_center_momentum
            if args.dino_center_momentum is not None
            else dino_cfg.get("center_momentum", 0.9)
        )
        dino_k = args.dino_k if args.dino_k is not None else dino_cfg.get("k", 1024)
        byol_cfg = pretrain_cfg.get("byol", {})
        byol_momentum = (
            args.byol_momentum
            if args.byol_momentum is not None
            else byol_cfg.get("momentum", 0.996)
        )
        task_cfg = DINOTaskConfig(
            data=data_cfg,
            optimizer=opt_cfg,
            out_dim=dino_k,
            tau_s=tau_s,
            tau_t=tau_t,
            center_momentum=center_momentum,
            ema_momentum=byol_momentum,
            save_full=args.save_full_pretrain
            if args.save_full_pretrain is not None
            else bool(pretrain_cfg.get("save_full_pretrain", False)),
        )

    training_cfg = pretrain_cfg.get("training", {})
    trainer_cfg = TrainerConfig(
        epochs=args.epochs or training_cfg.get("epochs", 200),
        amp=args.amp if args.amp is not None else bool(training_cfg.get("amp", False)),
        grad_clip_norm=args.grad_clip_norm
        if args.grad_clip_norm is not None
        else training_cfg.get("grad_clip_norm"),
        save_every=args.save_every or training_cfg.get("save_every", 1),
        save_dir=Path(args.save_dir or training_cfg.get("output_dir", "checkpoints")),
        log_file=Path(args.log_file)
        if args.log_file
        else (Path(training_cfg["log_file"]) if training_cfg.get("log_file") else None),
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
    parser.add_argument("--phi_checkpoint")
    parser.add_argument("--g_checkpoint")
    parser.add_argument("--d_checkpoint")
    parser.add_argument("--freeze_phi_epochs", type=int)
    parser.add_argument("--freeze_g_epochs", type=int)
    parser.add_argument("--freeze_d_epochs", type=int)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)

    parser.add_argument("--byol_momentum", type=float)
    parser.add_argument("--dino_tau_s", type=float)
    parser.add_argument("--dino_tau_t", type=float)
    parser.add_argument("--dino_center_momentum", type=float)
    parser.add_argument("--dino_k", type=int)
    parser.add_argument("--multi_crop", type=int)

    parser.add_argument("--crop_size", type=int)
    parser.add_argument("--local_crop_size", type=int)
    parser.add_argument("--max_translate", type=int)
    parser.add_argument("--hflip_prob", type=float)
    parser.add_argument("--vflip_prob", type=float)
    parser.add_argument("--rotate_prob", type=float)

    parser.add_argument("--blur_prob", type=float)
    parser.add_argument("--blur_kernel_min", type=int)
    parser.add_argument("--blur_kernel_max", type=int)
    parser.add_argument("--blur_sigma_min", type=float)
    parser.add_argument("--blur_sigma_max", type=float)
    parser.add_argument("--downsample_prob", type=float)
    parser.add_argument("--downsample_scale", type=int)
    parser.add_argument("--noise_std_min", type=float)
    parser.add_argument("--noise_std_max", type=float)
    parser.add_argument("--jitter_prob", type=float)
    parser.add_argument("--jitter_brightness", type=float)
    parser.add_argument("--jitter_contrast", type=float)

    parser.add_argument("--save_full_pretrain", action="store_true", default=None)

    args = parser.parse_args()
    if args.freeze_phi and args.freeze_phi_epochs is None:
        args.freeze_phi_epochs = -1
    if args.no_freeze_phi:
        args.freeze_phi_epochs = 0
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
