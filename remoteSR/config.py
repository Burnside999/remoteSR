from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


_DEFAULTS: dict[str, Any] = {
    "model": {
        "lr_channels": 3,
        "hr_channels": 3,
        "scale": 4,
        "num_feats": 96,
        "num_groups": 6,
        "blocks_per_group": 12,
        "ca_reduction": 16,
        "res_scale": 0.1,
        "kernel_size": 9,
        "share_kernel": False,
        "use_radiometric_affine": True,
        "distill_feat_channels": 64,
        "distill_feat_layers": 3,
        "distill_feat_downsample": 2,
        "align_window": 9,
        "align_temperature": 0.07,
        "align_normalize": True,
    },
    "data": {
        "train_lr_dir": "dataset/Himawari",
        "train_hr_dir": "dataset/Landsat",
        "scale": 4,
        "batch_size": 1,
        "num_workers": 4,
    },
    "optimizer": {
        "lr_g": 2e-4,
        "lr_d": 1e-4,
        "lr_phi": 2e-4,
        "weight_decay": 1e-4,
    },
    "training": {
        "epochs": 200,
        "amp": False,
        "grad_clip_norm": 1.0,
        "save_every": 10,
        "output_dir": "checkpoints",
        "onnx_export": {
            "enabled": False,
            "dir": "checkpoints/onnx",
            "opset": 18,
        },
        "log_file": "log.txt",
    },
    "loss": {
        "lambda_lr": 0.4,
        "lambda_match": 1.0,
        "lambda_ctx": 0.5,
        "lambda_tv": 0.0,
        "lambda_pix": 0.05,
        "lambda_grad": 0.1,
        "use_mask": False,
        "cx_bandwidth": 0.1,
        "cx_max_samples": 1024,
    },
    "evaluation": {
        "enabled": False,
        "lr_dir": "dataset/val_lr",
        "batch_size": 1,
        "num_workers": 2,
        "every": 1,
    },
    "inference": {
        "checkpoint": "good/nowindow64.pt",
        "input_dir": "testdata",
        "output_dir": "testoutput",
        "resize_factor": 0.5,
        "percentile_clip": [1, 99],
    },
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError(
                "pyyaml is required for YAML config files. Install via `pip install pyyaml`."
            )
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load a config file and merge with defaults.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _load_raw_config(cfg_path)
    merged = _deep_merge(_DEFAULTS, raw)
    return merged
