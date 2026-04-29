from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = ROOT / "src" / "generic_anomaly_pipeline.py"


PRESETS = {
    "wot_to_wot": {
        "train_path": ROOT / "data" / "processed_data" / "wot" / "wot_train_ml.parquet",
        "tune_path": ROOT / "data" / "processed_data" / "wot" / "wot_val_ml.parquet",
        "test_path": ROOT / "data" / "processed_data" / "wot" / "wot_val_ml.parquet",
        "output_model": ROOT / "models" / "wot_to_wot_anomaly.joblib",
        "output_report": ROOT / "reports" / "wot_to_wot_anomaly.json",
        "normal_label": 0,
    },
    "wot_to_dota": {
        "train_path": ROOT / "data" / "processed_data" / "wot" / "wot_train_ml.parquet",
        "tune_path": ROOT / "data" / "processed_data" / "wot" / "wot_val_ml.parquet",
        "test_path": ROOT / "data" / "processed_data" / "dota" / "dota_val_ml.parquet",
        "output_model": ROOT / "models" / "wot_to_dota_anomaly.joblib",
        "output_report": ROOT / "reports" / "wot_to_dota_anomaly.json",
        "normal_label": 0,
    },
    "dota_to_dota": {
        "train_path": ROOT / "data" / "processed_data" / "dota" / "dota_train_ml.parquet",
        "tune_path": ROOT / "data" / "processed_data" / "dota" / "dota_val_ml.parquet",
        "test_path": ROOT / "data" / "processed_data" / "dota" / "dota_val_ml.parquet",
        "output_model": ROOT / "models" / "dota_to_dota_anomaly.joblib",
        "output_report": ROOT / "reports" / "dota_to_dota_anomaly.json",
        "normal_label": 0,
    },
    "dota_to_wot": {
        "train_path": ROOT / "data" / "processed_data" / "dota" / "dota_train_ml.parquet",
        "tune_path": ROOT / "data" / "processed_data" / "dota" / "dota_val_ml.parquet",
        "test_path": ROOT / "data" / "processed_data" / "wot" / "wot_val_ml.parquet",
        "output_model": ROOT / "models" / "dota_to_wot_anomaly.joblib",
        "output_report": ROOT / "reports" / "dota_to_wot_anomaly.json",
        "normal_label": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed anomaly-detection experiment presets for the gaming toxicity datasets."
    )
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--threshold-quantile", type=float, default=0.26)
    parser.add_argument("--seed", type=int, default=7524)
    parser.add_argument("--text-col", default="clean_message")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--no-stopwords", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--train-path",
        str(preset["train_path"]),
        "--tune-path",
        str(preset["tune_path"]),
        "--test-path",
        str(preset["test_path"]),
        "--output-model",
        str(preset["output_model"]),
        "--output-report",
        str(preset["output_report"]),
        "--normal-label",
        str(preset["normal_label"]),
        "--threshold-quantile",
        str(args.threshold_quantile),
        "--seed",
        str(args.seed),
        "--text-col",
        args.text_col,
        "--label-col",
        args.label_col,
    ]

    if not args.no_stopwords:
        command.append("--use-custom-stopwords")

    print("Running preset:", args.preset)
    print("Command:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
