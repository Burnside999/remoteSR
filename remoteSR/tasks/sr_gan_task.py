from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from remoteSR.data import EvalLRDataset, SRDataset
from remoteSR.models import LossWeights, SemiSRConfig, SemiSRLoss, SemiSupervisedSRModel
from remoteSR.tasks.base import Task


@dataclass
class SROptimizerConfig:
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    lr_phi: float = 2e-4
    weight_decay: float = 1e-4


@dataclass
class SRLossConfig:
    lambda_lr: float = 0.4
    lambda_match: float = 1.0
    lambda_ctx: float = 0.5
    lambda_tv: float = 0.0
    lambda_pix: float = 0.05
    lambda_grad: float = 0.1
    lambda_texture: float = 0.1
    texture_kernel_size: int = 9
    texture_min_variance: float = 1e-3
    use_mask: bool = False
    cx_bandwidth: float = 0.1
    cx_max_samples: int = 1024


@dataclass
class SRDataConfig:
    train_lr_dir: str
    train_hr_dir: str
    scale: int = 4
    batch_size: int = 1
    num_workers: int = 4
    eval_enabled: bool = False
    eval_lr_dir: str | None = None
    eval_batch_size: int | None = None
    eval_num_workers: int | None = None


@dataclass
class SRTaskConfig:
    model: SemiSRConfig
    optimizer: SROptimizerConfig
    loss: SRLossConfig
    data: SRDataConfig
    phi_pretrained: str | None = None
    checkpoint_phi: str | None = None
    checkpoint_g: str | None = None
    checkpoint_d: str | None = None
    freeze_phi_epochs: int = 0
    freeze_g_epochs: int = 0
    freeze_d_epochs: int = 0


class SRGanTask(Task):
    metric_key = "loss_total"

    def __init__(self, config: SRTaskConfig) -> None:
        self.config = config
        self.criterion: SemiSRLoss | None = None
        self.frozen: dict[str, bool] = {"phi": False, "g": False, "d": False}

    def build_models(self, device: torch.device) -> dict[str, nn.Module]:
        model = SemiSupervisedSRModel(self.config.model).to(device)

        self._maybe_load_component(
            module=model.G,
            path=self.config.checkpoint_g,
            name="G",
            device=device,
        )
        self._maybe_load_component(
            module=model.D,
            path=self.config.checkpoint_d,
            name="D",
            device=device,
        )
        phi_loaded = self._maybe_load_component(
            module=model.phi,
            path=self.config.checkpoint_phi,
            name="phi",
            device=device,
        )
        if not phi_loaded and self.config.phi_pretrained:
            self._maybe_load_component(
                module=model.phi,
                path=self.config.phi_pretrained,
                name="phi",
                device=device,
            )

        weights = LossWeights(
            lambda_lr=self.config.loss.lambda_lr,
            lambda_match=self.config.loss.lambda_match,
            lambda_ctx=self.config.loss.lambda_ctx,
            lambda_tv=self.config.loss.lambda_tv,
            lambda_pix=self.config.loss.lambda_pix,
            lambda_grad=self.config.loss.lambda_grad,
            lambda_texture=self.config.loss.lambda_texture,
        )
        self.criterion = SemiSRLoss(
            model=model,
            weights=weights,
            cx_bandwidth=self.config.loss.cx_bandwidth,
            cx_max_samples=self.config.loss.cx_max_samples,
            use_mask=self.config.loss.use_mask,
            texture_kernel_size=self.config.loss.texture_kernel_size,
            texture_min_variance=self.config.loss.texture_min_variance,
        ).to(device)

        return {"sr_model": model}

    def build_optimizers(
        self, models: dict[str, nn.Module]
    ) -> dict[str, torch.optim.Optimizer]:
        model = models["sr_model"]
        opt_cfg = self.config.optimizer
        params = []
        if self.config.freeze_g_epochs != -1:
            params.append({"params": model.G.parameters(), "lr": opt_cfg.lr_g})
        if self.config.freeze_d_epochs != -1:
            params.append({"params": model.D.parameters(), "lr": opt_cfg.lr_d})
        if self.config.freeze_phi_epochs != -1:
            params.append({"params": model.phi.parameters(), "lr": opt_cfg.lr_phi})

        if not params:
            raise ValueError("All model components are frozen; nothing to optimize.")

        optimizer = torch.optim.AdamW(
            params, betas=(0.9, 0.99), weight_decay=opt_cfg.weight_decay
        )
        return {"optimizer": optimizer}

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        data_cfg = self.config.data
        train_loader = DataLoader(
            SRDataset(
                lr_dir=data_cfg.train_lr_dir,
                hr_dir=data_cfg.train_hr_dir,
                scale=data_cfg.scale,
            ),
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = None
        if data_cfg.eval_enabled and data_cfg.eval_lr_dir:
            val_loader = DataLoader(
                EvalLRDataset(lr_dir=data_cfg.eval_lr_dir),
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
        model = models["sr_model"]
        optimizer = optimizers["optimizer"]
        criterion = self.criterion
        assert criterion is not None

        model.train()
        self._apply_train_modes(model)
        batch = self._move_to_device(batch, device)

        y_lr = batch["y_lr"]
        x_ls_hr = batch["x_ls_hr"]
        mask_lr = batch.get("mask_lr")
        mask_hr = batch.get("mask_hr")
        x_hr_gt = batch.get("x_hr_gt")
        has_hr_gt = batch.get("has_hr_gt")

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

    @torch.no_grad()
    def val_step(
        self,
        batch: dict[str, Any],
        models: dict[str, nn.Module],
        device: torch.device,
    ) -> dict[str, float]:
        model = models["sr_model"]
        model.eval()
        batch = self._move_to_device(batch, device)
        y_lr = batch["y_lr"]
        x_hat, y_recon = model(y_lr, return_lr_recon=True)
        l1 = torch.nn.functional.l1_loss(y_recon, y_lr)
        return {"val_lr_l1": float(l1.item())}

    def config_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.config.model),
            "optimizer": asdict(self.config.optimizer),
            "loss": asdict(self.config.loss),
            "data": asdict(self.config.data),
            "phi_pretrained": self.config.phi_pretrained,
            "checkpoint_phi": self.config.checkpoint_phi,
            "checkpoint_g": self.config.checkpoint_g,
            "checkpoint_d": self.config.checkpoint_d,
            "freeze_phi_epochs": self.config.freeze_phi_epochs,
            "freeze_g_epochs": self.config.freeze_g_epochs,
            "freeze_d_epochs": self.config.freeze_d_epochs,
        }

    def on_epoch_start(
        self,
        epoch: int,
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> None:
        _ = optimizers  # optimizer behavior is gated by requires_grad
        model = models["sr_model"]

        self.frozen["g"] = self._should_freeze(self.config.freeze_g_epochs, epoch)
        self.frozen["d"] = self._should_freeze(self.config.freeze_d_epochs, epoch)
        self.frozen["phi"] = self._should_freeze(self.config.freeze_phi_epochs, epoch)

        self._set_requires_grad(model.G, not self.frozen["g"])
        self._set_requires_grad(model.D, not self.frozen["d"])
        self._set_requires_grad(model.phi, not self.frozen["phi"])

        # Keep frozen modules in eval mode to avoid BN stat updates.
        if self.frozen["g"]:
            model.G.eval()
        if self.frozen["d"]:
            model.D.eval()
        if self.frozen["phi"]:
            model.phi.eval()

    def save_component_checkpoints(
        self,
        save_dir: Path,
        epoch: int,
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        is_best: bool,
        routine_save: bool,
    ) -> None:
        _ = optimizers  # optimizer state is stored in the aggregated checkpoint
        model = models.get("sr_model")
        if model is None:
            return

        components = {"G": model.G, "D": model.D, "phi": model.phi}
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, module in components.items():
            payload = {"epoch": epoch, "state_dict": module.state_dict()}
            torch.save(payload, save_dir / f"{name.lower()}_latest.pth")
            if routine_save:
                torch.save(payload, save_dir / f"{name.lower()}_epoch_{epoch:04d}.pth")
            if is_best:
                torch.save(payload, save_dir / f"{name.lower()}_best.pth")

    @staticmethod
    def _should_freeze(freeze_epochs: int, epoch: int) -> bool:
        if freeze_epochs == -1:
            return True
        return epoch <= freeze_epochs

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for param in module.parameters():
            param.requires_grad = enabled

    def _apply_train_modes(self, model: SemiSupervisedSRModel) -> None:
        if hasattr(model, "G"):
            if self.frozen.get("g", False):
                model.G.eval()
            else:
                model.G.train()
        if hasattr(model, "D"):
            if self.frozen.get("d", False):
                model.D.eval()
            else:
                model.D.train()
        if hasattr(model, "phi"):
            if self.frozen.get("phi", False):
                model.phi.eval()
            else:
                model.phi.train()

    @staticmethod
    def _maybe_load_component(
        module: nn.Module,
        path: str | None,
        name: str,
        device: torch.device,
    ) -> bool:
        if not path:
            return False
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            print(f"{name} checkpoint not found: {ckpt_path}")
            return False

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "models" in state and isinstance(state["models"], dict):
                models_dict = state["models"]
                if name in models_dict and isinstance(models_dict[name], dict):
                    state = models_dict[name]
                elif "sr_model" in models_dict and isinstance(
                    models_dict["sr_model"], dict
                ):
                    sr_state = models_dict["sr_model"]
                    filtered = {
                        key.split(f"{name}.", 1)[1]: value
                        for key, value in sr_state.items()
                        if key.startswith(f"{name}.")
                    }
                    state = filtered or sr_state
            elif name in state and isinstance(state[name], dict):
                state = state[name]
        try:
            module.load_state_dict(state, strict=False)
            print(f"Loaded {name} weights from {ckpt_path}")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load {name} from {ckpt_path}: {exc}")
            return False

    @staticmethod
    def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
