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

        H, W, C = img.shape
        gsz = self.view_config.crop_size
        lsz = self.view_config.local_crop_size

        # --- 1) sample ONE global crop coord (anchor crop) ---
        if H < gsz or W < gsz:
            raise ValueError(f"Image too small for crop: {img.shape}, crop_size={gsz}")

        y0 = int(torch.randint(0, H - gsz + 1, (1,)).item())
        x0 = int(torch.randint(0, W - gsz + 1, (1,)).item())
        crop_g = img[y0 : y0 + gsz, x0 : x0 + gsz, :]  # (gsz,gsz,C)

        views: list[torch.Tensor] = []

        # --- 2) first two views: same crop, different stochastic aug/degrade ---
        for i in range(min(2, self.view_config.num_views)):
            view = self.pipeline.sample_view_from_crop(crop_g)
            view = torch.from_numpy(view).permute(2, 0, 1).float()
            views.append(view)

        # --- 3) local views: must be inside the global crop ---
        for i in range(2, self.view_config.num_views):
            if gsz < lsz:
                raise ValueError("local_crop_size should be <= crop_size")
            ly0 = int(torch.randint(0, gsz - lsz + 1, (1,)).item())
            lx0 = int(torch.randint(0, gsz - lsz + 1, (1,)).item())
            crop_l = crop_g[ly0 : ly0 + lsz, lx0 : lx0 + lsz, :]
            view = self.pipeline.sample_view_from_crop(crop_l)
            view = torch.from_numpy(view).permute(2, 0, 1).float()
            views.append(view)

        return {
            "views": views,
            "meta": {"file": self.hr_files[idx], "xywh": (x0, y0, gsz, gsz)},
        }
