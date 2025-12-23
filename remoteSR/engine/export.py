from __future__ import annotations

from pathlib import Path

import torch


def export_onnx_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    epoch: int,
    in_channels: int,
    opset: int = 18,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(
        1, in_channels, 224, 224, device=next(model.parameters()).device
    )
    torch.onnx.export(
        model,
        dummy_input,
        output_dir / f"model_epoch_{epoch}.onnx",
        opset_version=opset,
        input_names=["y_lr"],
        output_names=["output"],
        dynamic_axes={"y_lr": {0: "batch"}, "output": {0: "batch"}},
    )
