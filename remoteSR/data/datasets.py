from __future__ import annotations

import os
from typing import Dict

import cv2
import torch
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """
    Super-resolution training dataset that pairs LR/HR images by filename order.
    """
    def __init__(self, lr_dir: str, hr_dir: str, scale: int = 4):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.lr_files = os.listdir(lr_dir)
        self.hr_files = os.listdir(hr_dir)
        assert len(self.lr_files) == len(self.hr_files), "LR and HR directories must have the same number of files."

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr_img = cv2.imread(os.path.join(self.lr_dir, self.lr_files[idx]), cv2.IMREAD_UNCHANGED)
        hr_img = cv2.imread(os.path.join(self.hr_dir, self.hr_files[idx]), cv2.IMREAD_UNCHANGED)
        lr_img = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        hr_img = torch.from_numpy(hr_img).permute(2, 0, 1).float() / 255.0
        return {"y_lr": lr_img, "x_ls_hr": hr_img, "x_hr_gt": hr_img, "has_hr_gt": torch.tensor(1.0)}


class TestDataset(Dataset):
    """
    Simple test dataset that downsamples HR to synthesize LR for evaluation.
    """
    def __init__(self, lr_dir: str, hr_dir: str, scale: int = 4):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.lr_files = os.listdir(lr_dir)
        self.hr_files = os.listdir(hr_dir)
        assert len(self.lr_files) == len(self.hr_files), "LR and HR directories must have the same number of files."

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr_img = cv2.imread(os.path.join(self.hr_dir, self.hr_files[idx]), cv2.IMREAD_UNCHANGED)
        hr_img = torch.from_numpy(hr_img).permute(2, 0, 1).float() / 255.0
        lr_img = torch.nn.functional.interpolate(hr_img.unsqueeze(0), scale_factor=0.25, mode="area").squeeze(0)
        return {"y_lr": lr_img, "x_ls_hr": hr_img, "x_hr_gt": hr_img, "has_hr_gt": torch.tensor(1.0)}
    
class EvalLRDataset(Dataset):
    """
    Evaluation dataset that only uses LR images (no aligned HR required).
    """
    def __init__(self, lr_dir: str):
        self.lr_dir = lr_dir
        self.lr_files = os.listdir(lr_dir)

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        lr_img = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        lr_img = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        return {"y_lr": lr_img, "filename": os.path.basename(lr_path)}
