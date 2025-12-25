from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from remoteSR.tasks.base import Task


@dataclass
class TrainerConfig:
    epochs: int = 100
    amp: bool = False
    grad_clip_norm: float | None = 1.0
    save_every: int = 1
    save_dir: Path = Path("checkpoints")
    log_file: Path | None = None
    monitor: str = "loss"
    mode: str = "min"


class Trainer:
    """
    Generic training engine that is task-agnostic.

    The Task implementation provides the models, optimizers, and step logic.
    """

    def __init__(
        self,
        task: Task,
        config: TrainerConfig,
        device: torch.device,
        resume: Path | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.device = device
        self.resume = resume
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if config.amp and torch.cuda.is_available()
            else None
        )
        self.best_metric: float | None = None

    def build_dataloader(self) -> tuple[DataLoader, DataLoader | None]:
        return self.task.build_dataloaders()

    def log_metrics(self, epoch: int, metrics: dict[str, float], prefix: str) -> None:
        message = f"[{prefix}] epoch={epoch} " + ", ".join(
            f"{k}={v:.6f}" for k, v in metrics.items()
        )
        print(message)
        if self.config.log_file is not None:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.config.log_file.open("a", encoding="utf-8") as f:
                f.write(message + "\n")

    def _run_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        train: bool = True,
    ) -> dict[str, float]:
        running: dict[str, float] = {}
        count = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            if train:
                logs = self.task.train_step(
                    batch=batch,
                    models=self.models,
                    optimizers=self.optimizers,
                    scaler=self.scaler,
                    device=self.device,
                    amp=self.config.amp,
                    grad_clip_norm=self.config.grad_clip_norm,
                )
            else:
                logs = self.task.val_step(
                    batch=batch, models=self.models, device=self.device
                )
            count += 1
            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + value

        for key in running:
            running[key] /= max(1, count)
        return running

    def validate(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        return self.task.validate(
            epoch=epoch,
            dataloader=dataloader,
            models=self.models,
            device=self.device,
            default_val_step=self._run_epoch,
        )

    def save_ckpt(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
        routine_save: bool = False,
    ) -> None:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "optimizers": {k: opt.state_dict() for k, opt in self.optimizers.items()},
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "task_state": self.task.state_dict(),
            "config": self.task.config_dict(),
        }
        torch.save(ckpt, self.config.save_dir / "latest.pth")

        improved = False
        metric_source = val_metrics if val_metrics else train_metrics
        metric_key = self.task.metric_key
        metric_value = metric_source.get(metric_key)
        if metric_value is not None:
            if self.best_metric is None:
                improved = True
            else:
                improved = (
                    metric_value < self.best_metric
                    if self.config.mode == "min"
                    else metric_value > self.best_metric
                )
            if improved:
                self.best_metric = metric_value
                torch.save(ckpt, self.config.save_dir / "best.pth")
            if routine_save:
                torch.save(
                    ckpt,
                    self.config.save_dir / f"epoch_{epoch:04d}.pth",
                )

        self.task.save_component_checkpoints(
            save_dir=self.config.save_dir,
            epoch=epoch,
            models=self.models,
            optimizers=self.optimizers,
            is_best=improved,
            routine_save=routine_save,
        )

    def load_ckpt(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        for name, state in ckpt.get("models", {}).items():
            if name in self.models:
                self.models[name].load_state_dict(state)
        for name, state in ckpt.get("optimizers", {}).items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)
        if self.scaler is not None and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.task.load_state_dict(ckpt.get("task_state", {}))
        return int(ckpt.get("epoch", 0))

    def fit(self) -> None:
        train_loader, val_loader = self.build_dataloader()
        self.models = self.task.build_models(device=self.device)
        self.optimizers = self.task.build_optimizers(self.models)

        start_epoch = 1
        if self.resume is not None and self.resume.is_file():
            start_epoch = self.load_ckpt(self.resume) + 1

        for epoch in range(start_epoch, self.config.epochs + 1):
            self.task.on_epoch_start(
                epoch=epoch,
                models=self.models,
                optimizers=self.optimizers,
            )
            train_metrics = self._run_epoch(epoch, train_loader, train=True)
            self.log_metrics(epoch, train_metrics, prefix="train")

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(epoch, val_loader)
                if val_metrics:
                    self.log_metrics(epoch, val_metrics, prefix="val")

            if epoch % self.config.save_every == 0:
                self.save_ckpt(epoch, train_metrics, val_metrics, True)
            else:
                self.save_ckpt(epoch, train_metrics, val_metrics, False)

        self.task.on_train_end(self.models, self.config.save_dir)
