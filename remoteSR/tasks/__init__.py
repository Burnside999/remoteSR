from .base import Task
from .cyclegan_task import (
    CycleGANDataConfig,
    CycleGANLossConfig,
    CycleGANModelConfig,
    CycleGANOptimizerConfig,
    CycleGANTask,
    CycleGANTaskConfig,
)
from .phi_byol_task import BYOLDataConfig, BYOLOptimConfig, BYOLTaskConfig, PhiBYOLTask
from .phi_dino_task import DINODataConfig, DINOOptimConfig, DINOTaskConfig, PhiDINOTask
from .sr_gan_task import (
    SRDataConfig,
    SRGanTask,
    SRLossConfig,
    SROptimizerConfig,
    SRTaskConfig,
)

__all__ = [
    "Task",
    "BYOLDataConfig",
    "BYOLOptimConfig",
    "BYOLTaskConfig",
    "PhiBYOLTask",
    "DINODataConfig",
    "DINOOptimConfig",
    "DINOTaskConfig",
    "PhiDINOTask",
    "SRDataConfig",
    "SRLossConfig",
    "SROptimizerConfig",
    "SRTaskConfig",
    "SRGanTask",
]
