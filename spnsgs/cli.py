from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .paper import predict_paper, train_paper


def main() -> None:
    parser = argparse.ArgumentParser(description="SP-NSGS paper solver")
    parser.add_argument("action", choices=("train", "predict", "train_then_predict"))
    parser.add_argument("--config", default="config.paper.json")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config.training.output_dir) / "checkpoint_final.pt"
    )
    if args.action in {"train", "train_then_predict"}:
        checkpoint = train_paper(config)
        print(f"training complete: {checkpoint}")
    if args.action in {"predict", "train_then_predict"}:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"paper results: {predict_paper(config, checkpoint)}")
