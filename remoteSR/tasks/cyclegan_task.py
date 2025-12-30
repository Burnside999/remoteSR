from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from remoteSR.data import CycleGANDataset
from remoteSR.models import (
    CycleGANDiscriminator,
    CycleGANGenerator,
    CycleGANModelConfig,
)
from remoteSR.tasks.base import Task


@dataclass
class CycleGANDataConfig:
    domain_a_dir: str
    domain_b_dir: str
    batch_size: int = 1
    num_workers: int = 4
    normalize_tanh: bool = True
    eval_enabled: bool = False
    eval_domain_a_dir: str | None = None
    eval_domain_b_dir: str | None = None
    eval_batch_size: int | None = None
    eval_num_workers: int | None = None


@dataclass
class CycleGANOptimizerConfig:
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0


@dataclass
class CycleGANLossConfig:
    lambda_cycle: float = 10.0
    lambda_id: float = 0.5
    use_identity: bool = True


@dataclass
class CycleGANTaskConfig:
    model: CycleGANModelConfig
    data: CycleGANDataConfig
    optimizer: CycleGANOptimizerConfig
    loss: CycleGANLossConfig
    checkpoint_g_ab: str | None = None
    checkpoint_g_ba: str | None = None
    checkpoint_d_a: str | None = None
    checkpoint_d_b: str | None = None


class GANLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class CycleGANTask(Task):
    metric_key = "loss_g"

    def __init__(self, config: CycleGANTaskConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("remoteSR")
        self.criterion_gan = GANLoss()

    def build_models(self, device: torch.device) -> dict[str, nn.Module]:
        g_ab = CycleGANGenerator(self.config.model).to(device)
        g_ba = CycleGANGenerator(self.config.model).to(device)
        d_a = CycleGANDiscriminator(self.config.model).to(device)
        d_b = CycleGANDiscriminator(self.config.model).to(device)

        self._maybe_load_component(g_ab, self.config.checkpoint_g_ab, "G_AB", device)
        self._maybe_load_component(g_ba, self.config.checkpoint_g_ba, "G_BA", device)
        self._maybe_load_component(d_a, self.config.checkpoint_d_a, "D_A", device)
        self._maybe_load_component(d_b, self.config.checkpoint_d_b, "D_B", device)

        return {"G_AB": g_ab, "G_BA": g_ba, "D_A": d_a, "D_B": d_b}

    def build_optimizers(
        self, models: dict[str, nn.Module]
    ) -> dict[str, torch.optim.Optimizer]:
        opt_cfg = self.config.optimizer
        g_params = list(models["G_AB"].parameters()) + list(models["G_BA"].parameters())
        d_params = list(models["D_A"].parameters()) + list(models["D_B"].parameters())

        opt_g = torch.optim.Adam(
            g_params,
            lr=opt_cfg.lr_g,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            weight_decay=opt_cfg.weight_decay,
        )
        opt_d = torch.optim.Adam(
            d_params,
            lr=opt_cfg.lr_d,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            weight_decay=opt_cfg.weight_decay,
        )
        return {"optimizer_g": opt_g, "optimizer_d": opt_d}

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        data_cfg = self.config.data
        train_loader = DataLoader(
            CycleGANDataset(
                domain_a_dir=data_cfg.domain_a_dir,
                domain_b_dir=data_cfg.domain_b_dir,
                normalize_tanh=data_cfg.normalize_tanh,
            ),
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = None
        if (
            data_cfg.eval_enabled
            and data_cfg.eval_domain_a_dir
            and data_cfg.eval_domain_b_dir
        ):
            val_loader = DataLoader(
                CycleGANDataset(
                    domain_a_dir=data_cfg.eval_domain_a_dir,
                    domain_b_dir=data_cfg.eval_domain_b_dir,
                    normalize_tanh=data_cfg.normalize_tanh,
                ),
                batch_size=data_cfg.eval_batch_size or data_cfg.batch_size,
                shuffle=False,
                num_workers=data_cfg.eval_num_workers or data_cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        return train_loader, val_loader

    def train_step(
        self,
        batch: dict[str, Any],
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        scaler: torch.cuda.amp.GradScaler | None,
        device: torch.device,
        amp: bool,
        grad_clip_norm: float | None,
    ) -> dict[str, float]:
        g_ab = models["G_AB"]
        g_ba = models["G_BA"]
        d_a = models["D_A"]
        d_b = models["D_B"]
        opt_g = optimizers["optimizer_g"]
        opt_d = optimizers["optimizer_d"]

        g_ab.train()
        g_ba.train()
        d_a.train()
        d_b.train()

        batch = self._move_to_device(batch, device)
        real_a = batch["real_A"]
        real_b = batch["real_B"]

        self._set_requires_grad([d_a, d_b], False)
        opt_g.zero_grad(set_to_none=True)

        def _forward_g() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            fake_b = g_ab(real_a)
            rec_a = g_ba(fake_b)
            fake_a = g_ba(real_b)
            rec_b = g_ab(fake_a)

            loss_gan_ab = self.criterion_gan(d_b(fake_b), True)
            loss_gan_ba = self.criterion_gan(d_a(fake_a), True)
            loss_cycle = F.l1_loss(rec_a, real_a) + F.l1_loss(rec_b, real_b)

            loss_id = fake_b.new_tensor(0.0)
            if self.config.loss.use_identity and self.config.loss.lambda_id > 0:
                idt_a = g_ba(real_a)
                idt_b = g_ab(real_b)
                loss_id = F.l1_loss(idt_a, real_a) + F.l1_loss(idt_b, real_b)

            loss_g = (
                loss_gan_ab
                + loss_gan_ba
                + self.config.loss.lambda_cycle * loss_cycle
                + self.config.loss.lambda_id * loss_id
            )
            logs = {
                "loss_g": loss_g.detach(),
                "loss_gan_ab": loss_gan_ab.detach(),
                "loss_gan_ba": loss_gan_ba.detach(),
                "loss_cycle": loss_cycle.detach(),
                "loss_id": loss_id.detach(),
            }
            return loss_g, logs

        if scaler is None:
            loss_g, logs_g = _forward_g()
            loss_g.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(g_ab.parameters()) + list(g_ba.parameters()),
                    grad_clip_norm,
                )
            opt_g.step()
        else:
            with torch.amp.autocast("cuda", enabled=amp):
                loss_g, logs_g = _forward_g()
            scaler.scale(loss_g).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(
                    list(g_ab.parameters()) + list(g_ba.parameters()),
                    grad_clip_norm,
                )
            scaler.step(opt_g)
            scaler.update()

        self._set_requires_grad([d_a, d_b], True)
        opt_d.zero_grad(set_to_none=True)

        def _forward_d() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            with torch.no_grad():
                fake_b = g_ab(real_a)
                fake_a = g_ba(real_b)
            loss_d_a = 0.5 * (
                self.criterion_gan(d_a(real_a), True)
                + self.criterion_gan(d_a(fake_a.detach()), False)
            )
            loss_d_b = 0.5 * (
                self.criterion_gan(d_b(real_b), True)
                + self.criterion_gan(d_b(fake_b.detach()), False)
            )
            loss_d = loss_d_a + loss_d_b
            logs = {
                "loss_d": loss_d.detach(),
                "loss_d_a": loss_d_a.detach(),
                "loss_d_b": loss_d_b.detach(),
            }
            return loss_d, logs

        if scaler is None:
            loss_d, logs_d = _forward_d()
            loss_d.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(d_a.parameters()) + list(d_b.parameters()),
                    grad_clip_norm,
                )
            opt_d.step()
        else:
            with torch.amp.autocast("cuda", enabled=amp):
                loss_d, logs_d = _forward_d()
            scaler.scale(loss_d).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(
                    list(d_a.parameters()) + list(d_b.parameters()),
                    grad_clip_norm,
                )
            scaler.step(opt_d)
            scaler.update()

        logs = {**logs_g, **logs_d}
        return {k: float(v.item()) for k, v in logs.items()}

    @torch.no_grad()
    def val_step(
        self,
        batch: dict[str, Any],
        models: dict[str, nn.Module],
        device: torch.device,
    ) -> dict[str, float]:
        g_ab = models["G_AB"]
        g_ba = models["G_BA"]
        d_a = models["D_A"]
        d_b = models["D_B"]

        g_ab.eval()
        g_ba.eval()
        d_a.eval()
        d_b.eval()

        batch = self._move_to_device(batch, device)
        real_a = batch["real_A"]
        real_b = batch["real_B"]

        fake_b = g_ab(real_a)
        rec_a = g_ba(fake_b)
        fake_a = g_ba(real_b)
        rec_b = g_ab(fake_a)

        loss_gan_ab = self.criterion_gan(d_b(fake_b), True)
        loss_gan_ba = self.criterion_gan(d_a(fake_a), True)
        loss_cycle = F.l1_loss(rec_a, real_a) + F.l1_loss(rec_b, real_b)

        loss_id = fake_b.new_tensor(0.0)
        if self.config.loss.use_identity and self.config.loss.lambda_id > 0:
            idt_a = g_ba(real_a)
            idt_b = g_ab(real_b)
            loss_id = F.l1_loss(idt_a, real_a) + F.l1_loss(idt_b, real_b)

        loss_g = (
            loss_gan_ab
            + loss_gan_ba
            + self.config.loss.lambda_cycle * loss_cycle
            + self.config.loss.lambda_id * loss_id
        )

        loss_d_a = 0.5 * (
            self.criterion_gan(d_a(real_a), True)
            + self.criterion_gan(d_a(fake_a), False)
        )
        loss_d_b = 0.5 * (
            self.criterion_gan(d_b(real_b), True)
            + self.criterion_gan(d_b(fake_b), False)
        )
        loss_d = loss_d_a + loss_d_b

        logs = {
            "loss_g": float(loss_g.item()),
            "loss_d": float(loss_d.item()),
            "loss_gan_ab": float(loss_gan_ab.item()),
            "loss_gan_ba": float(loss_gan_ba.item()),
            "loss_cycle": float(loss_cycle.item()),
            "loss_id": float(loss_id.item()),
            "loss_d_a": float(loss_d_a.item()),
            "loss_d_b": float(loss_d_b.item()),
        }
        return logs

    def config_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.config.model),
            "data": asdict(self.config.data),
            "optimizer": asdict(self.config.optimizer),
            "loss": asdict(self.config.loss),
            "checkpoint_g_ab": self.config.checkpoint_g_ab,
            "checkpoint_g_ba": self.config.checkpoint_g_ba,
            "checkpoint_d_a": self.config.checkpoint_d_a,
            "checkpoint_d_b": self.config.checkpoint_d_b,
        }

    def save_component_checkpoints(
        self,
        save_dir: Path,
        epoch: int,
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        is_best: bool,
        routine_save: bool,
    ) -> None:
        _ = optimizers
        save_dir.mkdir(parents=True, exist_ok=True)
        components = {
            "G_AB": models.get("G_AB"),
            "G_BA": models.get("G_BA"),
            "D_A": models.get("D_A"),
            "D_B": models.get("D_B"),
        }
        for name, module in components.items():
            if module is None:
                continue
            payload = {"epoch": epoch, "state_dict": module.state_dict()}
            torch.save(payload, save_dir / f"{name.lower()}_latest.pth")
            if routine_save:
                torch.save(payload, save_dir / f"{name.lower()}_epoch_{epoch:04d}.pth")
            if is_best:
                torch.save(payload, save_dir / f"{name.lower()}_best.pth")

    def _maybe_load_component(
        self,
        module: nn.Module,
        path: str | None,
        name: str,
        device: torch.device,
    ) -> bool:
        if not path:
            return False
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            self.logger.warning("%s checkpoint not found: %s", name, ckpt_path)
            return False

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "models" in state and isinstance(state["models"], dict):
                models_dict = state["models"]
                if name in models_dict and isinstance(models_dict[name], dict):
                    state = models_dict[name]
        try:
            module.load_state_dict(state, strict=False)
            self.logger.info("Loaded %s weights from %s", name, ckpt_path)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to load %s from %s: %s", name, ckpt_path, exc)
            return False

    @staticmethod
    def _set_requires_grad(modules: list[nn.Module], enabled: bool) -> None:
        for module in modules:
            for param in module.parameters():
                param.requires_grad = enabled

    @staticmethod
    def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
