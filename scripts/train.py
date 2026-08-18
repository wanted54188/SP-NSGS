"""Train one public SP-NSGS or ISO configuration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnsgs.config import load_config
from spnsgs.paper import train_paper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Full or ISO JSON configuration")
    parser.add_argument("--resume", action="store_true", help="resume from output_dir/last_checkpoint.pt when present")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = Path(config.training.output_dir) / "last_checkpoint.pt"
    result = train_paper(config, resume_checkpoint=checkpoint if args.resume and checkpoint.exists() else None)
    print(f"training complete: {result}")


if __name__ == "__main__":
    main()
