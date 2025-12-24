from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


class Task(ABC):
    """
    Base Task interface for the training engine.
    """

    metric_key: str = "loss"

    @abstractmethod
    def build_models(self, device: torch.device) -> dict[str, nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def build_optimizers(
        self, models: dict[str, nn.Module]
    ) -> dict[str, torch.optim.Optimizer]:
        raise NotImplementedError

    @abstractmethod
    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def val_step(
        self,
        batch: dict[str, Any],
        models: dict[str, nn.Module],
        device: torch.device,
    ) -> dict[str, float]:
        return {}

    def validate(
        self,
        epoch: int,
        dataloader: DataLoader,
        models: dict[str, nn.Module],
        device: torch.device,
        default_val_step: Callable[[int, DataLoader, bool], dict[str, float]],
    ) -> dict[str, float]:
        _ = epoch
        _ = models
        _ = device
        return default_val_step(epoch, dataloader, train=False)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state

    def config_dict(self) -> dict[str, Any]:
        return {}

    def on_train_end(self, models: dict[str, nn.Module], save_dir: Path) -> None:
        _ = models
        _ = save_dir
