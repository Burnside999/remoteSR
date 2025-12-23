import argparse
import torch.multiprocessing as mp

from remoteSR.infer import run_inference
from remoteSR.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="remoteSR entrypoint")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file (yaml/json)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run training")

    infer_p = subparsers.add_parser("infer", help="Run inference")
    infer_p.add_argument("--input_dir", help="Override input directory for inference")
    infer_p.add_argument("--checkpoint", help="Override checkpoint path")
    infer_p.add_argument("--output_dir", help="Override output directory for inference")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        run_training(args.config)
    elif args.command == "infer":
        run_inference(
            config_path=args.config,
            input_dir=args.input_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
