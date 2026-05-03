#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, list_model_configs, list_models


DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_METHODS = ["regular", "stratified", "train_only_nontoxic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Common training-evaluation entry point used by runners."
    )
    parser.add_argument(
        "--model-task",
        choices=["supervised", "unsupervised", "bert"],
        required=True,
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model-configs", default="all")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--text-col", default="message")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--normal-label", type=int, default=0)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--seed", type=int, default=7524)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def supervised_command(args: argparse.Namespace) -> list[str]:
    models = args.models or ["all"]
    output_csv = args.output_dir / "supervised.csv"
    command = [
        sys.executable,
        "src/run_training_evals.py",
        "--split-root",
        str(args.split_root),
        "--output-csv",
        str(output_csv),
        "--methods",
        *args.methods,
        "--models",
        *models,
        "--model-configs",
        args.model_configs,
        "--text-col",
        args.text_col,
        "--label-col",
        args.label_col,
        "--normal-label",
        str(args.normal_label),
        "--seed",
        str(args.seed),
    ]
    if args.binary:
        command.append("--binary")
    if args.oversample:
        command.append("--oversample")
    return command


def unsupervised_commands(args: argparse.Namespace) -> list[list[str]]:
    models = args.models or ["all"]
    output_csv = args.output_dir / "unsupervised.csv"
    return [
        [
            sys.executable,
            "src/run_training_evals.py",
            "--split-root",
            str(args.split_root),
            "--output-csv",
            str(output_csv),
            "--methods",
            "train_only_nontoxic",
            "--models",
            *models,
            "--model-configs",
            args.model_configs,
            "--text-col",
            args.text_col,
            "--label-col",
            args.label_col,
            "--normal-label",
            str(args.normal_label),
            "--seed",
            str(args.seed),
        ]
    ]


def bert_commands(args: argparse.Namespace) -> list[list[str]]:
    models = args.models or ["bert"]
    commands = []
    for model_name in models:
        config_names = list_model_configs(model_name) if args.model_configs == "all" else [args.model_configs]
        for config_name in config_names:
            spec = build_model(model_name, config_name, seed=args.seed).build()
            for method in args.methods:
                method_dir = args.split_root / method
                output_dir = PROJECT_ROOT / "models" / f"{model_name}_{config_name}_{method}"
                commands.append(
                    spec.to_train_command(
                        train_data=method_dir / "train.parquet",
                        val_data=method_dir / "val.parquet",
                        output_dir=output_dir,
                        text_col=args.text_col,
                        label_col=args.label_col,
                    )
                )
    return commands


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_task == "supervised":
        commands = [supervised_command(args)]
    elif args.model_task == "unsupervised":
        commands = unsupervised_commands(args)
    else:
        commands = bert_commands(args)

    for command in commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
