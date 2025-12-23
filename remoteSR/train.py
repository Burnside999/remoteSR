from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from remoteSR.config import load_config
from remoteSR.data import EvalLRDataset, SRDataset
from remoteSR.models import LossWeights, SemiSRConfig, SemiSRLoss, SemiSupervisedSRModel
from remoteSR.utils.training import build_optimizer, evaluate_lr_reconstruction, train_one_epoch


def _maybe_export_onnx(model: torch.nn.Module, output_dir: Path, epoch: int, in_channels: int, opset: int = 18) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, in_channels, 224, 224, device=next(model.parameters()).device)
    torch.onnx.export(
        model,
        dummy_input,
        output_dir / f"model_epoch_{epoch}.onnx",
        opset_version=opset,
        input_names=["y_lr"],
        output_names=["output"],
        dynamic_axes={"y_lr": {0: "batch"}, "output": {0: "batch"}},
    )


def run_training(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    train_loader = DataLoader(
        SRDataset(lr_dir=data_cfg["train_lr_dir"], hr_dir=data_cfg["train_hr_dir"], scale=data_cfg["scale"]),
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    eval_loader = None
    eval_cfg = cfg.get("evaluation", {})
    eval_every = int(eval_cfg.get("every", 1))
    if eval_cfg.get("enabled", False):
        eval_lr_dir = eval_cfg.get("lr_dir")
        if eval_lr_dir and Path(eval_lr_dir).is_dir():
            eval_loader = DataLoader(
                EvalLRDataset(lr_dir=eval_lr_dir),
                batch_size=eval_cfg.get("batch_size", data_cfg["batch_size"]),
                shuffle=False,
                num_workers=eval_cfg.get("num_workers", data_cfg["num_workers"]),
                pin_memory=device.type == "cuda",
            )
        else:
            print(f"[Eval] Skipping evaluation: lr_dir not found ({eval_lr_dir})")

    model_cfg = cfg["model"]
    model = SemiSupervisedSRModel(SemiSRConfig(**model_cfg)).to(device)

    opt_cfg = cfg["optimizer"]
    optimizer = build_optimizer(
        model,
        lr_g=opt_cfg["lr_g"],
        lr_d=opt_cfg["lr_d"],
        lr_phi=opt_cfg["lr_phi"],
        weight_decay=opt_cfg["weight_decay"],
    )

    loss_cfg = cfg["loss"]
    weights = LossWeights(
        lambda_lr=loss_cfg["lambda_lr"],
        lambda_match=loss_cfg["lambda_match"],
        lambda_ctx=loss_cfg["lambda_ctx"],
        lambda_tv=loss_cfg["lambda_tv"],
        lambda_pix=loss_cfg["lambda_pix"],
        lambda_grad=loss_cfg["lambda_grad"],
    )
    criterion = SemiSRLoss(
        model=model,
        weights=weights,
        cx_bandwidth=loss_cfg["cx_bandwidth"],
        cx_max_samples=loss_cfg["cx_max_samples"],
        use_mask=loss_cfg.get("use_mask", False),
    ).to(device)

    train_cfg = cfg["training"]
    use_amp = bool(train_cfg.get("amp", False))
    scaler = torch.amp.GradScaler("cuda") if use_amp and torch.cuda.is_available() else None

    log_file = Path(train_cfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        logs: Dict[str, float] = train_one_epoch(
            epoch=epoch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            scaler=scaler,
            amp=use_amp,
            grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        )

        if eval_loader is not None and (epoch % eval_every == 0):
            eval_logs = evaluate_lr_reconstruction(
                model=model,
                dataloader=eval_loader,
                device=device,
                desc=f"Eval {epoch}",
            )

        print(epoch, logs)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"epoch: {epoch}, {logs}\n")

        if epoch % train_cfg["save_every"] == 0:
            ckpt_dir = Path(train_cfg["output_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_dir / f"model_epoch_{epoch}.pt",
            )

            onnx_cfg = train_cfg.get("onnx_export", {})
            if onnx_cfg.get("enabled", False):
                _maybe_export_onnx(
                    model,
                    Path(onnx_cfg.get("dir", "onnx")),
                    epoch,
                    in_channels=model_cfg["lr_channels"],
                    opset=onnx_cfg.get("opset", 18),
                )


if __name__ == "__main__":
    run_training()
