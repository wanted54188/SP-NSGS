"""One isolated formal SP-NSGS experiment executed by the batch runner."""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnsgs.config import load_config
from spnsgs.paper import predict_paper, train_paper


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-checkpoint")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    config = load_config(args.config)
    status = {
        "experiment_id": args.experiment_id,
        "experiment_type": "training",
        "model_variant": config.paper.experiment_variant,
        "sensor_fraction": config.paper.train_sensor_fraction,
        "seed": config.training.seed,
        "status": "running",
        "current_epoch": 0,
        "total_epochs": config.paper.epochs,
        "start_time": _now(),
        "last_update_time": _now(),
        "end_time": None,
        "last_checkpoint": str(run_dir / "last_checkpoint.pt"),
        "best_composite_checkpoint": str(run_dir / "best_composite.pt"),
        "error_message": None,
        "process_exit_code": None,
    }
    _write(status_path, status)

    def update(row: dict) -> None:
        status["current_epoch"] = int(row["epoch"])
        status["last_update_time"] = _now()
        _write(status_path, status)

    with (run_dir / "run.log").open("a", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            checkpoint = run_dir / "last_checkpoint.pt"
            if args.evaluate_checkpoint:
                result = Path(args.evaluate_checkpoint)
            else:
                result = train_paper(
                    config,
                    resume_checkpoint=checkpoint if args.resume and checkpoint.exists() else None,
                    status_callback=update,
                )
            predict_paper(config, result)
            summary_path = run_dir / "predictions" / "quantitative_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(summary) or ["status"])
                writer.writeheader()
                writer.writerow(summary or {"status": "completed"})
            status.update(status="completed", current_epoch=config.paper.epochs, end_time=_now(), process_exit_code=0)
            _write(status_path, status)
            return 0
        except Exception:
            status.update(status="failed", end_time=_now(), error_message=traceback.format_exc(), process_exit_code=1)
            _write(status_path, status)
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
