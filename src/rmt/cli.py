"""Command-line interface for validation, experiments, and reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rmt.config import TrainingSettings, load_yaml
from rmt.data.loader import group_records, load_records
from rmt.data.validator import validate_dataset
from rmt.methods.ewc import EWCStrategy
from rmt.methods.experience_replay import ExperienceReplay
from rmt.methods.joint_training import JointTraining
from rmt.methods.sequential_lora import SequentialLoRA
from rmt.models.lora import LoRASettings
from rmt.models.qwen import initialize_qwen
from rmt.reporting.example_data import generate_example_package
from rmt.reporting.figures import generate_all_figures
from rmt.training.runner import ExperimentRunner, RunSpec, create_run_schedule
from rmt.training.trainer import TrainingEngine

ROOT = Path(__file__).resolve().parents[2]


def _execute_run(specification: RunSpec) -> dict:
    experiment = load_yaml(ROOT / "configs/experiment.yaml")
    model_config = load_yaml(ROOT / "configs/model.yaml")
    methods = load_yaml(ROOT / "configs/methods.yaml")
    records = load_records(ROOT / experiment["dataset_path"])
    datasets = group_records(records)
    bundle = initialize_qwen(
        model_config["model_name"],
        model_config["use_fp16_on_cuda"],
        lora_settings=LoRASettings.from_mapping(model_config["lora"]),
    )
    engine = TrainingEngine(
        bundle,
        TrainingSettings.from_mapping(experiment["training"]),
        model_config["maximum_sequence_length"],
        model_config["text_truncation_tokens"],
    )
    if specification.method == "sequential_lora":
        strategy = SequentialLoRA()
    elif specification.method == "experience_replay":
        fraction = specification.parameters.get(
            "buffer_fraction", methods["experience_replay"]["buffer_fraction"]
        )
        strategy = ExperienceReplay(float(fraction))
    elif specification.method == "ewc":
        value = specification.parameters.get("lambda", methods["ewc"]["lambda"])
        strategy = EWCStrategy(
            float(value),
            engine.loss,
            int(methods["ewc"]["fisher_samples"]),
            float(methods["ewc"]["online_gamma"]),
        )
    elif specification.method == "joint_training":
        strategy = JointTraining()
    else:
        raise ValueError(f"Unknown method {specification.method}")
    result = strategy.run(
        bundle.model,
        datasets,
        experiment["task_order"],
        specification.seed,
        engine.train,
        engine.evaluate,
    )
    return result.to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rmt")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument(
        "--manifest", type=Path, default=ROOT / "data/manifests/assignment4_dataset_manifest.json"
    )
    commands.add_parser("schedule")
    examples = commands.add_parser("generate-examples")
    examples.add_argument("--output", type=Path, default=ROOT / "examples/results")
    figures = commands.add_parser("generate-figures")
    figures.add_argument("--input", type=Path, default=ROOT / "examples/results")
    figures.add_argument("--output", type=Path, default=ROOT / "examples/figures")
    run = commands.add_parser("run")
    run.add_argument("--run-id", help="Run one schedule entry. Omit with --full")
    run.add_argument("--full", action="store_true", help="Execute all 19 planned runs")
    run.add_argument("--output", type=Path, default=ROOT / "results/raw")
    run.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        report = validate_dataset(args.dataset, args.manifest)
        print(json.dumps(asdict(report), indent=2))
        return 0 if report.valid else 1
    if args.command == "schedule":
        print(json.dumps([asdict(run) for run in create_run_schedule()], indent=2))
        return 0
    if args.command == "generate-examples":
        print(json.dumps(generate_example_package(args.output), indent=2))
        return 0
    if args.command == "generate-figures":
        print("\n".join(str(path) for path in generate_all_figures(args.input, args.output)))
        return 0
    schedule = create_run_schedule()
    if not args.full:
        if not args.run_id:
            raise SystemExit("Specify --run-id or --full")
        schedule = [run for run in schedule if run.run_id == args.run_id]
        if not schedule:
            raise SystemExit(f"Run ID not found: {args.run_id}")
    runner = ExperimentRunner(args.output)
    results = runner.run(schedule, _execute_run, resume=not args.no_resume)
    return 0 if all(result["status"] == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

