from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate_prob: float = 0.5
    max_translate: int = 4


@dataclass
class DegradationConfig:
    blur_prob: float = 0.8
    blur_kernel_min: int = 3
    blur_kernel_max: int = 7
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0
    downsample_prob: float = 0.0
    downsample_scale: int = 4
    noise_std_min: float = 0.0
    noise_std_max: float = 0.02
    jitter_prob: float = 0.8
    jitter_brightness: float = 0.1
    jitter_contrast: float = 0.1


class AugmentPipeline:
    def __init__(
        self,
        augment: AugmentConfig,
        degradation: DegradationConfig,
    ) -> None:
        self.augment = augment
        self.degradation = degradation

    def _random_crop(self, img: np.ndarray, crop_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            img = cv2.resize(img, (max(w, crop_size), max(h, crop_size)))
            h, w = img.shape[:2]
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        return img[top : top + crop_size, left : left + crop_size]

    def _random_flip_rotate(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.augment.hflip_prob:
            img = cv2.flip(img, 1)
        if np.random.rand() < self.augment.vflip_prob:
            img = cv2.flip(img, 0)
        if np.random.rand() < self.augment.rotate_prob:
            k = np.random.randint(0, 4)
            img = np.rot90(img, k)
        return np.ascontiguousarray(img)

    def _random_translate(self, img: np.ndarray) -> np.ndarray:
        max_t = self.augment.max_translate
        if max_t <= 0:
            return img
        tx = np.random.randint(-max_t, max_t + 1)
        ty = np.random.randint(-max_t, max_t + 1)
        if tx == 0 and ty == 0:
            return img
        h, w = img.shape[:2]
        mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.degradation.blur_prob:
            kmin = self.degradation.blur_kernel_min
            kmax = self.degradation.blur_kernel_max
            kernel = np.random.randint(kmin, kmax + 1)
            if kernel % 2 == 0:
                kernel += 1
            sigma = np.random.uniform(
                self.degradation.blur_sigma_min, self.degradation.blur_sigma_max
            )
            img = cv2.GaussianBlur(img, (kernel, kernel), sigma)

        if np.random.rand() < self.degradation.downsample_prob:
            scale = max(1, int(self.degradation.downsample_scale))
            if scale > 1:
                h, w = img.shape[:2]
                img = cv2.resize(
                    img, (w // scale, h // scale), interpolation=cv2.INTER_AREA
                )
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        noise_std = np.random.uniform(
            self.degradation.noise_std_min, self.degradation.noise_std_max
        )
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=img.shape).astype(np.float32)
            img = img + noise

        if np.random.rand() < self.degradation.jitter_prob:
            brightness = np.random.uniform(
                -self.degradation.jitter_brightness, self.degradation.jitter_brightness
            )
            contrast = 1.0 + np.random.uniform(
                -self.degradation.jitter_contrast, self.degradation.jitter_contrast
            )
            img = img * contrast + brightness

        return np.clip(img, 0.0, 1.0)

    def sample_view_from_crop(self, crop: np.ndarray) -> np.ndarray:
        view = crop
        view = self._random_flip_rotate(view)
        view = self._random_translate(view)
        view = self._degrade(view)
        return view

    def sample_view(self, img: np.ndarray, crop_size: int) -> np.ndarray:
        crop = self._random_crop(img, crop_size)
        return self.sample_view_from_crop(crop)
