from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import torch
from torch.utils.data import Dataset

from remoteSR.data.augmentations import (
    AugmentConfig,
    AugmentPipeline,
    DegradationConfig,
)


@dataclass
class PretrainViewConfig:
    crop_size: int
    local_crop_size: int
    num_views: int


class PretrainPairDataset(Dataset):
    """
    Dataset that returns multiple augmented views for self-supervised pretraining.
    """

    def __init__(
        self,
        hr_dir: str,
        crop_size: int,
        local_crop_size: int,
        num_views: int,
        augment: AugmentConfig,
        degradation: DegradationConfig,
    ) -> None:
        self.hr_dir = hr_dir
        self.hr_files = os.listdir(hr_dir)
        self.view_config = PretrainViewConfig(crop_size, local_crop_size, num_views)
        self.pipeline = AugmentPipeline(augment=augment, degradation=degradation)

    def __len__(self) -> int:
        return len(self.hr_files)

    def _read_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.astype("float32") / 255.0
        return img

    def __getitem__(self, idx: int) -> dict[str, list[torch.Tensor]]:
        path = os.path.join(self.hr_dir, self.hr_files[idx])
        img = self._read_image(path)

        views = []
        for i in range(self.view_config.num_views):
            crop_size = (
                self.view_config.crop_size
                if i < 2
                else self.view_config.local_crop_size
            )
            view = self.pipeline.sample_view(img, crop_size)
            view = torch.from_numpy(view).permute(2, 0, 1).float()
            views.append(view)

        return {"views": views}
