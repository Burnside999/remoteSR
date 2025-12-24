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
class BYOLDataConfig:
    hr_dir: str
    batch_size: int = 16
    num_workers: int = 4
    crop_size: int = 96
    local_crop_size: int = 64
    num_views: int = 2
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)


@dataclass
class BYOLOptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class BYOLTaskConfig:
    data: BYOLDataConfig
    optimizer: BYOLOptimConfig
    in_channels: int = 3
    projector_dim: int = 256
    byol_momentum: float = 0.996
    save_full: bool = False


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _update_ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    # EMA update for parameters
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.data.copy_(s_buf.data)


class PhiBYOLTask(Task):
    metric_key = "loss"

    def __init__(self, config: BYOLTaskConfig) -> None:
        self.config = config

    def build_models(self, device: torch.device) -> dict[str, nn.Module]:
        student_phi = TinySwinPhi(in_channels=self.config.in_channels).to(device)
        teacher_phi = TinySwinPhi(in_channels=self.config.in_channels).to(device)
        projector = MLP(64, self.config.projector_dim, self.config.projector_dim).to(
            device
        )
        teacher_projector = MLP(
            64, self.config.projector_dim, self.config.projector_dim
        ).to(device)
        predictor = Predictor(self.config.projector_dim, self.config.projector_dim).to(
            device
        )

        teacher_phi.load_state_dict(student_phi.state_dict())
        teacher_projector.load_state_dict(projector.state_dict())
        for param in teacher_phi.parameters():
            param.requires_grad = False
        for param in teacher_projector.parameters():
            param.requires_grad = False

        return {
            "phi_student": student_phi,
            "phi_teacher": teacher_phi,
            "projector": projector,
            "projector_teacher": teacher_projector,
            "predictor": predictor,
        }

    def build_optimizers(
        self, models: dict[str, nn.Module]
    ) -> dict[str, torch.optim.Optimizer]:
        params = list(models["phi_student"].parameters())
        params += list(models["projector"].parameters())
        params += list(models["predictor"].parameters())
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
        projector = models["projector"]
        teacher_projector = models["projector_teacher"]
        predictor = models["predictor"]
        optimizer = optimizers["optimizer"]

        student_phi.train()
        projector.train()
        predictor.train()
        teacher_phi.eval()
        teacher_projector.eval()

        views = batch["views"]
        x1 = views[0].to(device, non_blocking=True)
        x2 = views[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        def _forward_student(x: torch.Tensor):
            feats = student_phi(x)  # (B,64,H,W)
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # (B,64)
            proj = projector(pooled)  # (B,D)
            pred = predictor(proj)  # (B,D)
            return pred, pooled, proj

        def _forward_teacher(x: torch.Tensor):
            feats = teacher_phi(x)
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            proj = teacher_projector(pooled)
            return proj, pooled

        if scaler is None:
            p1, f1, z1s = _forward_student(x1)
            p2, f2, z2s = _forward_student(x2)
            with torch.no_grad():
                z1, tf1 = _forward_teacher(x1)  # teacher proj
                z2, tf2 = _forward_teacher(x2)

            loss = 2 - 2 * F.cosine_similarity(p1, z2.detach(), dim=1).mean()
            loss += 2 - 2 * F.cosine_similarity(p2, z1.detach(), dim=1).mean()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(student_phi.parameters())
                    + list(projector.parameters())
                    + list(predictor.parameters()),
                    grad_clip_norm,
                )
            optimizer.step()
        else:
            with torch.amp.autocast("cuda", enabled=amp):
                p1, f1, z1s = _forward_student(x1)
                p2, f2, z2s = _forward_student(x2)
                with torch.no_grad():
                    z1, tf1 = _forward_teacher(x1)
                    z2, tf2 = _forward_teacher(x2)

                loss = 2 - 2 * F.cosine_similarity(p1, z2.detach(), dim=1).mean()
                loss += 2 - 2 * F.cosine_similarity(p2, z1.detach(), dim=1).mean()

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student_phi.parameters())
                    + list(projector.parameters())
                    + list(predictor.parameters()),
                    grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()

        _update_ema(teacher_phi, student_phi, self.config.byol_momentum)
        _update_ema(teacher_projector, projector, self.config.byol_momentum)

        with torch.no_grad():
            p = F.normalize(p1.float(), dim=1)
            z = F.normalize(z2.float(), dim=1)
            sim = p @ z.t()  # (B,B)
            top1 = sim.argmax(dim=1)
            top1_acc = (
                (top1 == torch.arange(sim.size(0), device=sim.device)).float().mean()
            )

            # sanity: are two views actually different?
            x12_l1 = (x1 - x2).abs().mean()

            # cosine stats (pos/neg)
            pos_cos = F.cosine_similarity(p1.float(), z2.float(), dim=1).mean()
            perm = torch.randperm(z2.size(0), device=z2.device)
            neg_cos = F.cosine_similarity(p1.float(), z2.float()[perm], dim=1).mean()
            cos_gap = pos_cos - neg_cos

            # backbone embedding stats (THIS is what you really care about)
            feat = torch.cat([f1.float(), f2.float()], dim=0)  # (2B,64)
            feat_var = feat.var(dim=0).mean()
            feat_std = feat.std(dim=0).mean()
            feat_norm = feat.norm(dim=1).mean()

            # projector stats (student)
            proj = torch.cat([z1s.float(), z2s.float()], dim=0)  # (2B,D)
            proj_var = proj.var(dim=0).mean()
            proj_std = proj.std(dim=0).mean()
            proj_norm = proj.norm(dim=1).mean()

            # predictor stats (student) - your old emb_var was basically this
            pred = torch.cat([p1.float(), p2.float()], dim=0)
            pred_var = pred.var(dim=0).mean()
            pred_std = pred.std(dim=0).mean()
            pred_norm = pred.norm(dim=1).mean()

        return {
            "loss": float(loss.item()),
            "top1_acc": float(top1_acc.item()),
            "x12_l1": float(x12_l1.item()),
            "pos_cos": float(pos_cos.item()),
            "neg_cos": float(neg_cos.item()),
            "cos_gap": float(cos_gap.item()),
            "feat_var": float(feat_var.item()),
            "feat_std": float(feat_std.item()),
            "feat_norm": float(feat_norm.item()),
            "proj_var": float(proj_var.item()),
            "proj_std": float(proj_std.item()),
            "proj_norm": float(proj_norm.item()),
            "pred_var": float(pred_var.item()),
            "pred_std": float(pred_std.item()),
            "pred_norm": float(pred_norm.item()),
        }

    def on_train_end(self, models: dict[str, nn.Module], save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        phi_state = models["phi_student"].state_dict()
        torch.save(phi_state, save_dir / "phi_backbone.pth")
        if self.config.save_full:
            torch.save(
                {
                    "phi": phi_state,
                    "projector": models["projector"].state_dict(),
                    "predictor": models["predictor"].state_dict(),
                },
                save_dir / "phi_full_pretrain.pth",
            )

    def config_dict(self) -> dict[str, Any]:
        return {
            "data": asdict(self.config.data),
            "optimizer": asdict(self.config.optimizer),
            "in_channels": self.config.in_channels,
            "projector_dim": self.config.projector_dim,
            "byol_momentum": self.config.byol_momentum,
            "save_full": self.config.save_full,
        }
