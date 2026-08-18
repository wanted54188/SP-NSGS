"""Evaluate a frozen or newly trained SP-NSGS checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnsgs.config import load_config
from spnsgs.paper import predict_paper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = predict_paper(config, checkpoint)
    print(f"evaluation complete: {output}")


if __name__ == "__main__":
    main()
