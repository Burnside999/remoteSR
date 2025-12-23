from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F

__all__ = ["TinySwinPhi", "build_shift_mask"]


def _pad_to_window(x, window_size: int):
    # x: (B,C,H,W) -> pad H,W to multiples of window_size
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, H, W)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w, H, W)

def _unpad(x, meta):
    pad_h, pad_w, H, W = meta
    return x[..., :H, :W]

def window_partition(x, window_size: int):
    # x: (B,H,W,C) -> (num_windows*B, win, win, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    # windows: (num_windows*B, win, win, C) -> (B,H,W,C)
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, attn_mask=None):
        # x: (Bwin, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, heads, N, N)
        if attn_mask is not None:
            # attn_mask: (num_windows, N, N) broadcast to (B_, heads, N, N)
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1)  # (.., nW, heads, N, N)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B_, N, C)
        return self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)

    def forward(self, x, H, W, attn_mask=None):
        # x: (B, H*W, C)
        B, L, C = x.shape
        assert L == H * W
        shortcut = x

        x = self.norm1(x).view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))

        # partition windows
        x_windows = window_partition(x, self.window_size)  # (nW*B, win, win, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

def build_shift_mask(H, W, window_size, shift_size, device):
    # standard Swin attention mask for shifted windows
    img_mask = torch.zeros((1, H, W, 1), device=device)  # (1,H,W,1)
    cnt = 0
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size*window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class TinySwinPhi(nn.Module):
    """
    phi(.) feature extractor:
    input : (B, C_in, H, W)
    output: (B, C_feat, H, W)  (same spatial size, transformer features)
    """
    def __init__(self, in_channels=3, embed_dim=64, depth=4, num_heads=4, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.embed = nn.Conv2d(in_channels, embed_dim, 1, bias=True)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(SwinBlock(embed_dim, num_heads=num_heads, window_size=window_size, shift_size=shift, mlp_ratio=mlp_ratio))
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=True)

    def forward(self, x):
        # x: (B,C,H,W)
        x = self.embed(x)
        x, meta = _pad_to_window(x, self.window_size)
        B, C, H, W = x.shape

        # tokens
        t = x.permute(0,2,3,1).contiguous().view(B, H*W, C)

        attn_mask = None
        # build mask only if shifted blocks exist
        for blk in self.blocks:
            if blk.shift_size > 0:
                attn_mask = build_shift_mask(H, W, self.window_size, blk.shift_size, x.device)
                break

        for blk in self.blocks:
            t = blk(t, H, W, attn_mask=attn_mask if blk.shift_size > 0 else None)

        out = t.view(B, H, W, C).permute(0,3,1,2).contiguous()
        out = self.proj(out)
        out = _unpad(out, meta)
        return out
