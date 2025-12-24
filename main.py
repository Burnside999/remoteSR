import argparse
import sys

import torch.multiprocessing as mp

from remoteSR.infer import run_inference
from remoteSR.train import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="remoteSR entrypoint")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run training")

    infer_p = subparsers.add_parser("infer", help="Run inference")
    infer_p.add_argument("--config", default="config/default.yaml")
    infer_p.add_argument("--input_dir", help="Override input directory for inference")
    infer_p.add_argument("--checkpoint", help="Override checkpoint path")
    infer_p.add_argument("--output_dir", help="Override output directory for inference")

    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()

    if args.command == "train":
        sys.argv = [sys.argv[0]] + remaining
        train_main()
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
