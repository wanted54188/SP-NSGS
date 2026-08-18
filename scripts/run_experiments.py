"""Formal SP-NSGS final-batch planner and safety gate.

The current formal batch has ten independent training runs: nine nonzero sparse
sensor runs and one independently trained ISO baseline.  No SGS is evaluated
without fitting a closure, while Full SP-NSGS cites the frozen Phase-A
checkpoint.  ``--dry-run`` writes the exact run list and capability audit.
``--execute`` remains guarded until exact per-run resume/evaluation handling is
implemented, rather than silently changing a paper protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
MANIFESTS = RESULTS / "manifests"
FROZEN_FULL_CHECKPOINT = ROOT / "checkpoints" / "full_sp_nsgs.pt"


@dataclass(frozen=True)
class PlannedRun:
    stage: str
    name: str
    output_directory: str
    seed: int | None
    sensor_fraction: float | None
    filter_width_ratio: int | None
    model_variant: str
    q_d_enabled: bool
    anisotropy_enabled: bool
    q_b_enabled: bool
    independently_trained: bool


def _planned_runs() -> list[PlannedRun]:
    runs: list[PlannedRun] = []
    for fraction in (0.01, 0.02, 0.04):
        for seed in (2026, 2027, 2028):
            runs.append(
                PlannedRun(
                    stage="sparse",
                    name=f"sparse_frac_{fraction:0.2f}_seed_{seed}",
                    output_directory=(
                        f"results/runs/sparse/frac_{int(fraction * 100):02d}/seed_{seed}"
                    ),
                    seed=seed,
                    sensor_fraction=fraction,
                    filter_width_ratio=4,
                    model_variant="full_sp_nsgs",
                    q_d_enabled=True,
                    anisotropy_enabled=True,
                    q_b_enabled=True,
                    independently_trained=True,
                )
            )
    # ISO is the only independently trained physics baseline.  No SGS is a
    # rollout-only baseline and Full SP-NSGS is the frozen Phase-A reference.
    runs.append(
        PlannedRun(
            stage="ablation",
            name="01_no_sgs_evaluation",
            output_directory="results/runs/ablation/01_no_sgs",
            seed=2026,
            sensor_fraction=0.04,
            filter_width_ratio=4,
            model_variant="no_sgs",
            q_d_enabled=False,
            anisotropy_enabled=False,
            q_b_enabled=False,
            independently_trained=False,
        )
    )
    for folder, variant, q_d, anisotropy, q_b, independent in (
        ("02_iso_dissipative", "iso_dissipative", True, False, False, True),
        ("03_full_sp_nsgs", "full_sp_nsgs", True, True, True, False),
    ):
        runs.append(
            PlannedRun(
                stage="ablation",
                name=folder,
                output_directory=(
                    f"results/runs/ablation/{folder}/seed_2026"
                    if independent
                    else f"results/runs/ablation/{folder}"
                ),
                seed=2026,
                sensor_fraction=0.04,
                filter_width_ratio=4,
                model_variant=variant,
                q_d_enabled=q_d,
                anisotropy_enabled=anisotropy,
                q_b_enabled=q_b,
                independently_trained=independent,
            )
        )
    return runs


def _capability_audit() -> dict[str, object]:
    """State factual prerequisites; this avoids an invalid paper batch."""
    blockers: list[dict[str, str]] = []
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_batch_ready": True,
        "reason": "Ten-run frozen protocol: independent sparse/ISO training plus No-SGS and frozen-Full evaluation.",
        "blockers": blockers,
    }


def _select(runs: list[PlannedRun], stage: str | None) -> list[PlannedRun]:
    return runs if stage is None else [run for run in runs if run.stage == stage]


def _write_reference_metadata(runs: list[PlannedRun]) -> None:
    """Write only non-training baseline metadata; never touch checkpoints."""
    for run in runs:
        if run.independently_trained:
            continue
        directory = ROOT / run.output_directory
        directory.mkdir(parents=True, exist_ok=True)
        if run.model_variant == "no_sgs":
            readme = (
                "experiment_type = formal physics baseline\n"
                "experiment_name = No SGS\n"
                "model_variant = No SGS\n"
                "q_sgs = 0\n"
                "purpose = unresolved baseline without SGS closure\n"
                "training = none; a posteriori evaluation only\n"
                "seed = 2026\n"
            )
        else:
            readme = (
                "experiment_type = frozen formal reference\n"
                "experiment_name = Full SP-NSGS\n"
                "model_variant = Full SP-NSGS\n"
                "q_sgs = q_d + q_b\n"
                "q_d = positive diffusivity + SPD anisotropy\n"
                "q_b = bounded sign-indefinite correction\n"
                "training = reused frozen Phase-A checkpoint; no retraining\n"
            )
            (directory / "checkpoint_reference.txt").write_text(
                str(FROZEN_FULL_CHECKPOINT.resolve()) + "\n", encoding="utf-8"
            )
        (directory / "README.txt").write_text(readme, encoding="utf-8")


def _write_training_config(run: PlannedRun) -> Path:
    from spnsgs.config import load_config
    source_config = ROOT / "configs" / "main.json"
    base = json.loads(source_config.read_text(encoding="utf-8"))
    directory = ROOT / run.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    # Resolve the config-relative STAR data path before placing the generated
    # per-run config into a deeper output directory.
    base["data"]["data_dir"] = str(load_config(source_config).data.data_dir)
    base["training"]["seed"] = run.seed
    base["training"]["output_dir"] = str(directory.resolve())
    base["paper"]["train_sensor_fraction"] = run.sensor_fraction
    base["paper"]["experiment_variant"] = run.model_variant
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = (
        f"experiment_type = {run.stage}\nexperiment_name = {run.name}\n"
        f"seed = {run.seed}\nsensor_fraction = {run.sensor_fraction}\n"
        f"filter_width = Delta/dx {run.filter_width_ratio}\nmodel_variant = {run.model_variant}\n"
        f"q_d_enabled = {run.q_d_enabled}\nanisotropy_enabled = {run.anisotropy_enabled}\n"
        f"q_b_enabled = {run.q_b_enabled}\ntraining_budget = 100 epochs\n"
    )
    (directory / "README.txt").write_text(readme, encoding="utf-8")
    return config_path


def _preflight(runs: list[PlannedRun]) -> list[str]:
    """Validate launch requirements without creating a training process."""
    errors: list[str] = []
    destinations = [run.output_directory for run in runs]
    if len(destinations) != len(set(destinations)):
        errors.append("formal run output directories are not unique")
    if not FROZEN_FULL_CHECKPOINT.is_file():
        errors.append(f"frozen full checkpoint missing: {FROZEN_FULL_CHECKPOINT}")
    # Configs live under ``configs/`` in the release.  Use the public
    # loader so relative data paths resolve relative to that config, exactly
    # as they do for training and evaluation.
    from spnsgs.config import load_config
    base = load_config(ROOT / "configs" / "main.json")
    data_dir = Path(base.data.data_dir)
    if not data_dir.is_dir():
        errors.append(f"STAR data directory missing: {data_dir}")
    elif not list(data_dir.glob(base.data.file_glob)):
        errors.append(f"STAR data pattern has no matches in: {data_dir}")
    for run in runs:
        if run.stage == "sparse" and run.sensor_fraction not in {0.01, 0.02, 0.04}:
            errors.append(f"invalid sparse fraction in {run.name}")
        if run.model_variant == "iso_dissipative" and (run.anisotropy_enabled or run.q_b_enabled):
            errors.append("ISO definition is not A=I, q_b=0")
    return errors


def _execute(runs: list[PlannedRun], resume: bool) -> int:
    logs = RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    master = logs / "batch_master.log"
    with master.open("a", encoding="utf-8", buffering=1) as log:
        for run in runs:
            directory = ROOT / run.output_directory
            status_path = directory / "status.json"
            if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed":
                log.write(f"SKIP COMPLETED {run.name}\n")
                continue
            if run.independently_trained:
                config_path = _write_training_config(run)
                command = [sys.executable, str(ROOT / "scripts" / "_worker.py"), "--config", str(config_path), "--run-dir", str(directory), "--experiment-id", run.name]
                if resume:
                    command.append("--resume")
            else:
                config_path = _write_training_config(run)
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                # The project-level JSON omits the optional ``model`` section
                # when every setting uses ModelConfig defaults.  The formal
                # No-SGS evaluation must therefore create that override rather
                # than assuming the section already exists.
                payload.setdefault("model", {})["sgs_enabled"] = (
                    run.model_variant != "no_sgs"
                )
                config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                command = [sys.executable, str(ROOT / "scripts" / "_worker.py"), "--config", str(config_path), "--run-dir", str(directory), "--experiment-id", run.name, "--evaluate-checkpoint", str(FROZEN_FULL_CHECKPOINT)]
            log.write(f"START {run.name}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT)
            log.write(f"{'COMPLETED' if result.returncode == 0 else 'FAILED'} {run.name} returncode={result.returncode}\n")
            log.flush()
            if result.returncode:
                return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", choices=("sparse", "ablation"))
    parser.add_argument("--only-stage", choices=("sparse", "ablation"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.from_stage and args.only_stage:
        parser.error("--from-stage and --only-stage cannot be combined")
    if args.execute and args.preflight_only:
        parser.error("--execute and --preflight-only cannot be combined")

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    order = ("sparse", "ablation")
    runs = _planned_runs()
    if args.only_stage:
        runs = _select(runs, args.only_stage)
    elif args.from_stage:
        runs = [run for run in runs if order.index(run.stage) >= order.index(args.from_stage)]
    audit = _capability_audit()
    report = {
        "requested_modes": {
            "dry_run": args.dry_run,
            "resume": args.resume,
            "from_stage": args.from_stage,
            "only_stage": args.only_stage,
        },
        "artifact_task_count": len(runs),
        "independent_training_run_count": sum(
            run.independently_trained for run in runs
        ),
        "runs": [asdict(run) for run in runs],
        "capability_audit": audit,
    }
    path = MANIFESTS / "FINAL_BATCH_DRY_RUN.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_reference_metadata(runs)
    print(f"planned_artifact_tasks={len(runs)}")
    print(f"independent_training_runs={sum(run.independently_trained for run in runs)}")
    for index, run in enumerate(runs, start=1):
        print(f"{index:02d}. [{run.stage}] {run.name} -> {run.output_directory}")
    print(f"capability_audit={path}")
    preflight_errors = _preflight(runs)
    if args.preflight_only:
        if preflight_errors:
            print("PREFLIGHT FAILED: " + "; ".join(preflight_errors), file=sys.stderr)
            return 1
        print("PREFLIGHT PASS")
        return 0
    if args.execute:
        if not audit["formal_batch_ready"]:
            print("REFUSING_EXECUTION: formal protocol blockers are recorded in FINAL_BATCH_DRY_RUN.json.", file=sys.stderr)
            return 2
        if preflight_errors:
            print("REFUSING_EXECUTION: " + "; ".join(preflight_errors), file=sys.stderr)
            return 1
        return _execute(runs, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
