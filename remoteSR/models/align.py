from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftShiftAligner(nn.Module):
    """
    Local soft matching for misalignment-robust distillation.

    Given query features Q and key features K (same shape B,C,H,W),
    for each pixel p in Q, we look for the best matching location in a local window in K
    using a softmax over similarities, then output aligned teacher features T.

    This is differentiable and does NOT require explicit sub-pixel registration.

    Implementation uses unfold:
      - K patches: (B, C, win^2, H*W)
      - similarity: dot(Q, K_patch) -> (B, win^2, H*W)
      - softmax over win^2 -> weights
      - aligned feature: weighted sum of K_patch -> (B, C, H, W)

    ⚠️ Memory note:
      unfold allocates (B, C*win^2, H*W), so please use patches in training.
    """

    def __init__(
        self,
        window_size: int = 9,
        temperature: float = 0.07,
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert window_size % 2 == 1, "window_size should be odd."
        self.window_size = window_size
        self.temperature = float(temperature)
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, C, H, W) from prediction
            key:   (B, C, H, W) from teacher (e.g., Landsat)
            return_weights: if True, also return weights map (B, win^2, H, W)

        Returns:
            aligned_key: (B, C, H, W)
            (optional) weights: (B, win^2, H, W)
        """
        assert query.shape == key.shape, f"query {query.shape} != key {key.shape}"
        B, C, H, W = query.shape
        win = self.window_size
        pad = win // 2

        if self.normalize:
            query_n = F.normalize(query, dim=1, eps=self.eps)
            key_n = F.normalize(key, dim=1, eps=self.eps)
        else:
            query_n = query
            key_n = key

        # Extract local patches from key: (B, C*win*win, H*W)
        key_patches = F.unfold(key_n, kernel_size=win, padding=pad)  # (B, C*K, HW)
        K = win * win
        key_patches = key_patches.view(B, C, K, H * W)  # (B, C, K, HW)

        q = query_n.view(B, C, H * W)  # (B, C, HW)

        # Similarity: dot product over C -> (B, K, HW)
        sim = (key_patches * q.unsqueeze(2)).sum(dim=1)  # (B, K, HW)
        sim = sim / self.temperature

        weights = F.softmax(sim, dim=1)  # (B, K, HW)

        # Weighted sum -> aligned key features: (B, C, HW) -> (B, C, H, W)
        aligned = (key_patches * weights.unsqueeze(1)).sum(dim=2)  # (B, C, HW)
        aligned = aligned.view(B, C, H, W)

        if return_weights:
            weights_map = weights.view(B, K, H, W)
            return aligned, weights_map
        return aligned

    def distill_l1(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        detach_key: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Convenience: compute L1 distillation loss with local soft alignment.
        Typically you want detach_key=True (teacher stopgrad).
        """
        if detach_key:
            key = key.detach()
        aligned = self.forward(query, key, return_weights=False)
        loss = F.l1_loss(query, aligned, reduction=reduction)
        return loss
