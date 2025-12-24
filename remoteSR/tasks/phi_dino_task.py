from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from remoteSR.data import PretrainPairDataset
from remoteSR.data.augmentations import AugmentConfig, DegradationConfig
from remoteSR.models import TinySwinPhi
from remoteSR.tasks.base import Task


@dataclass
class DINODataConfig:
    hr_dir: str
    batch_size: int = 16
    num_workers: int = 4
    crop_size: int = 96
    local_crop_size: int = 64
    num_views: int = 2
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)


@dataclass
class DINOOptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class DINOTaskConfig:
    data: DINODataConfig
    optimizer: DINOOptimConfig
    in_channels: int = 3
    head_dim: int = 256
    out_dim: int = 1024
    tau_s: float = 0.1
    tau_t: float = 0.04
    center_momentum: float = 0.9
    ema_momentum: float = 0.996
    save_full: bool = False


class DinoHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _update_ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


class PhiDINOTask(Task):
    metric_key = "loss"

    def __init__(self, config: DINOTaskConfig) -> None:
        self.config = config
        self.center: torch.Tensor | None = None

    def build_models(self, device: torch.device) -> dict[str, nn.Module]:
        student_phi = TinySwinPhi(in_channels=self.config.in_channels).to(device)
        teacher_phi = TinySwinPhi(in_channels=self.config.in_channels).to(device)
        head_student = DinoHead(64, self.config.head_dim, self.config.out_dim).to(
            device
        )
        head_teacher = DinoHead(64, self.config.head_dim, self.config.out_dim).to(
            device
        )

        teacher_phi.load_state_dict(student_phi.state_dict())
        head_teacher.load_state_dict(head_student.state_dict())
        for param in teacher_phi.parameters():
            param.requires_grad = False
        for param in head_teacher.parameters():
            param.requires_grad = False

        if self.center is None:
            self.center = torch.zeros(1, self.config.out_dim, device=device)
        else:
            self.center = self.center.to(device)

        return {
            "phi_student": student_phi,
            "phi_teacher": teacher_phi,
            "head_student": head_student,
            "head_teacher": head_teacher,
        }

    def build_optimizers(
        self, models: dict[str, nn.Module]
    ) -> dict[str, torch.optim.Optimizer]:
        params = list(models["phi_student"].parameters())
        params += list(models["head_student"].parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        return {"optimizer": optimizer}

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        data_cfg = self.config.data
        dataset = PretrainPairDataset(
            hr_dir=data_cfg.hr_dir,
            crop_size=data_cfg.crop_size,
            local_crop_size=data_cfg.local_crop_size,
            num_views=data_cfg.num_views,
            augment=data_cfg.augment,
            degradation=data_cfg.degradation,
        )
        loader = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return loader, None

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
        student_phi = models["phi_student"]
        teacher_phi = models["phi_teacher"]
        head_student = models["head_student"]
        head_teacher = models["head_teacher"]
        optimizer = optimizers["optimizer"]
        assert self.center is not None

        student_phi.train()
        head_student.train()
        teacher_phi.eval()
        head_teacher.eval()

        views = batch["views"]
        x1 = views[0].to(device, non_blocking=True)
        x2 = views[1].to(device, non_blocking=True)
        extra_views = [v.to(device, non_blocking=True) for v in views[2:]]

        optimizer.zero_grad(set_to_none=True)

        def _forward_student(x: torch.Tensor) -> torch.Tensor:
            feats = student_phi(x)
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            return head_student(pooled)

        def _forward_teacher(x: torch.Tensor) -> torch.Tensor:
            feats = teacher_phi(x)
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            return head_teacher(pooled)

        if scaler is None:
            s1 = _forward_student(x1)
            s2 = _forward_student(x2)
            s_locals = [_forward_student(v) for v in extra_views]
            with torch.no_grad():
                t1 = _forward_teacher(x1)
                t2 = _forward_teacher(x2)
            loss = self._dino_loss(s1, t2) + self._dino_loss(s2, t1)
            for s_local in s_locals:
                loss = (
                    loss + self._dino_loss(s_local, t1) + self._dino_loss(s_local, t2)
                )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(student_phi.parameters()) + list(head_student.parameters()),
                    grad_clip_norm,
                )
            optimizer.step()
        else:
            with torch.amp.autocast("cuda", enabled=amp):
                s1 = _forward_student(x1)
                s2 = _forward_student(x2)
                s_locals = [_forward_student(v) for v in extra_views]
                with torch.no_grad():
                    t1 = _forward_teacher(x1)
                    t2 = _forward_teacher(x2)
                loss = self._dino_loss(s1, t2) + self._dino_loss(s2, t1)
                for s_local in s_locals:
                    loss = (
                        loss
                        + self._dino_loss(s_local, t1)
                        + self._dino_loss(s_local, t2)
                    )

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student_phi.parameters()) + list(head_student.parameters()),
                    grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            t_all = torch.cat([t1, t2], dim=0)
            batch_center = t_all.mean(dim=0, keepdim=True)
            self.center.mul_(self.config.center_momentum).add_(
                batch_center * (1.0 - self.config.center_momentum)
            )

        _update_ema(teacher_phi, student_phi, self.config.ema_momentum)
        _update_ema(head_teacher, head_student, self.config.ema_momentum)

        emb_tensors = [s1.detach(), s2.detach()] + [s.detach() for s in s_locals]
        emb_var = torch.cat(emb_tensors, dim=0).var(dim=0).mean()
        pos_cos = F.cosine_similarity(s1.detach(), t2.detach(), dim=1).mean()

        return {
            "loss": float(loss.item()),
            "emb_var": float(emb_var.item()),
            "pos_cos": float(pos_cos.item()),
        }

    def _dino_loss(
        self, student_out: torch.Tensor, teacher_out: torch.Tensor
    ) -> torch.Tensor:
        assert self.center is not None
        teacher_probs = F.softmax(
            (teacher_out - self.center) / self.config.tau_t, dim=1
        )
        student_log_probs = F.log_softmax(student_out / self.config.tau_s, dim=1)
        return -(teacher_probs * student_log_probs).sum(dim=1).mean()

    def state_dict(self) -> dict[str, Any]:
        return {"center": self.center}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        center = state.get("center")
        if center is not None:
            self.center = center

    def on_train_end(self, models: dict[str, nn.Module], save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        phi_state = models["phi_student"].state_dict()
        torch.save(phi_state, save_dir / "phi_backbone.pth")
        if self.config.save_full:
            torch.save(
                {
                    "phi": phi_state,
                    "head": models["head_student"].state_dict(),
                    "center": self.center,
                },
                save_dir / "phi_full_pretrain.pth",
            )

    def config_dict(self) -> dict[str, Any]:
        return {
            "data": asdict(self.config.data),
            "optimizer": asdict(self.config.optimizer),
            "in_channels": self.config.in_channels,
            "head_dim": self.config.head_dim,
            "out_dim": self.config.out_dim,
            "tau_s": self.config.tau_s,
            "tau_t": self.config.tau_t,
            "center_momentum": self.config.center_momentum,
            "ema_momentum": self.config.ema_momentum,
            "save_full": self.config.save_full,
        }
