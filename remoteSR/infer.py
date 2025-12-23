from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from remoteSR.config import load_config
from remoteSR.models import SemiSRConfig, SemiSupervisedSRModel


def _load_image(
    path: Path, resize_factor: float, device: torch.device
) -> torch.Tensor | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_factor and resize_factor != 1.0:
        new_h = max(1, int(img.shape[0] * resize_factor))
        new_w = max(1, int(img.shape[1] * resize_factor))
        img = cv2.resize(img, (new_w, new_h))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def run_inference(
    config_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inf_cfg = dict(cfg["inference"])
    if input_dir:
        inf_cfg["input_dir"] = str(input_dir)
    if checkpoint:
        inf_cfg["checkpoint"] = str(checkpoint)
    if output_dir:
        inf_cfg["output_dir"] = str(output_dir)

    model_cfg = cfg["model"]
    model = SemiSupervisedSRModel(SemiSRConfig(**model_cfg)).to(device)

    ckpt_path = Path(inf_cfg["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    input_root = Path(inf_cfg["input_dir"])
    output_root = Path(inf_cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    resize_factor = float(inf_cfg.get("resize_factor", 1.0))
    percentile_clip = inf_cfg.get("percentile_clip", [1, 99])
    lo_p, hi_p = percentile_clip if len(percentile_clip) == 2 else (1, 99)

    for img_path in sorted([p for p in input_root.iterdir() if p.is_file()]):
        tensor = _load_image(img_path, resize_factor, device)
        if tensor is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        with torch.no_grad():
            output = model(tensor)

        out = output.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        lo = np.percentile(out, lo_p)
        hi = np.percentile(out, hi_p)
        out = (out - lo) / (hi - lo + 1e-8)
        out = np.clip(out, 0, 1)

        out_u8 = (out * 255.0).round().astype(np.uint8)
        out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_root / img_path.name), out_bgr)


if __name__ == "__main__":
    run_inference()
