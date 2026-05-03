from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split


SplitMethod = Literal[
    "regular",
    "stratified",
    "train_only_nontoxic",
    "normal_train_mixed_eval",
    "all",
    "both",
]

DEFAULT_INPUT_ROOT = Path("data/processed_data")
DEFAULT_OUTPUT_ROOT = Path("data/splits")
DEFAULT_SEED = 7524


@dataclass(frozen=True)
class SplitConfig:
    inputs: list[Path]
    output_root: Path
    method: SplitMethod
    label_col: str
    normal_label: int
    val_size: float
    test_size: float
    seed: int
    stratify_source: bool


def parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/validation/test splits from processed gaming-toxicity data. "
            "Supported methods: regular random sampling, standard stratified sampling, "
            "and train-only-nontoxic splits where validation/test remain mixed."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Input .parquet or .csv files. Defaults to full processed dataset files under "
            "data/processed_data, excluding existing *_train_ml and *_val_ml splits."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--method",
        choices=["regular", "stratified", "train_only_nontoxic", "normal_train_mixed_eval", "all", "both"],
        default="all",
        help="Which split method to write.",
    )
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--normal-label", type=int, default=0)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--stratify-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stratify by both source dataset and label when possible. Use --no-stratify-source to stratify by label only.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be less than 1.")

    inputs = args.inputs if args.inputs is not None else discover_default_inputs(DEFAULT_INPUT_ROOT)
    if not inputs:
        raise ValueError("No input files found. Pass --inputs explicitly.")

    return SplitConfig(
        inputs=inputs,
        output_root=args.output_root,
        method=args.method,
        label_col=args.label_col,
        normal_label=args.normal_label,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        stratify_source=args.stratify_source,
    )


def discover_default_inputs(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.glob("*/*.parquet")):
        if path.name.endswith(("_train_ml.parquet", "_val_ml.parquet")):
            continue
        paths.append(path)
    return paths


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type for {path}. Expected .parquet or .csv.")


def load_inputs(paths: list[Path], label_col: str, seed: int) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = read_table(path).copy()
        if label_col not in df.columns:
            raise ValueError(f"{path} does not contain label column {label_col!r}.")
        df[label_col] = df[label_col].astype(int)
        df["source_dataset"] = path.parent.name
        df["source_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratify_key(df: pd.DataFrame, config: SplitConfig) -> pd.Series:
    if config.stratify_source:
        return df["source_dataset"].astype(str) + "__" + df[config.label_col].astype(str)
    return df[config.label_col]


def safe_stratify(df: pd.DataFrame, key: pd.Series) -> pd.Series | None:
    counts = key.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    return key


def split_regular(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    holdout_size = config.val_size + config.test_size
    train, holdout = train_test_split(
        df,
        test_size=holdout_size,
        random_state=config.seed,
        shuffle=True,
    )
    relative_test_size = config.test_size / holdout_size
    val, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        random_state=config.seed,
        shuffle=True,
    )
    return reset_splits({"train": train, "val": val, "test": test})


def split_stratified(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    holdout_size = config.val_size + config.test_size
    train, holdout = train_test_split(
        df,
        test_size=holdout_size,
        random_state=config.seed,
        stratify=safe_stratify(df, stratify_key(df, config)),
    )
    relative_test_size = config.test_size / holdout_size
    val, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        random_state=config.seed,
        stratify=safe_stratify(holdout, stratify_key(holdout, config)),
    )
    return reset_splits({"train": train, "val": val, "test": test})


def split_train_only_nontoxic_val_test_mixed(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    normal = df.loc[df[config.label_col] == config.normal_label]
    non_normal = df.loc[df[config.label_col] != config.normal_label]

    holdout_size = config.val_size + config.test_size
    normal_train, normal_holdout = train_test_split(
        normal,
        test_size=holdout_size,
        random_state=config.seed,
        stratify=safe_stratify(normal, normal["source_dataset"]) if config.stratify_source else None,
    )
    normal_val, normal_test = split_eval_pool(normal_holdout, config, stratify_labels=False)
    anomaly_val, anomaly_test = split_eval_pool(non_normal, config, stratify_labels=True)

    val = pd.concat([normal_val, anomaly_val], ignore_index=True)
    test = pd.concat([normal_test, anomaly_test], ignore_index=True)

    splits = {
        "train": normal_train,
        "val": val.sample(frac=1.0, random_state=config.seed),
        "test": test.sample(frac=1.0, random_state=config.seed),
    }
    return reset_splits(splits)


def split_normal_train_mixed_eval(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    return split_train_only_nontoxic_val_test_mixed(df, config)


def split_train_only_nontoxic(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    return split_train_only_nontoxic_val_test_mixed(df, config)


def split_eval_pool(df: pd.DataFrame, config: SplitConfig, stratify_labels: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    relative_test_size = config.test_size / (config.val_size + config.test_size)
    key = None
    if stratify_labels:
        key = stratify_key(df, config)
    elif config.stratify_source:
        key = df["source_dataset"]

    val, test = train_test_split(
        df,
        test_size=relative_test_size,
        random_state=config.seed,
        stratify=safe_stratify(df, key) if key is not None else None,
    )
    return val, test


def reset_splits(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: split.reset_index(drop=True) for name, split in splits.items()}


def summarize_split(df: pd.DataFrame, config: SplitConfig) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "label_counts": stringify_keys(df[config.label_col].value_counts().sort_index().to_dict()),
        "source_counts": stringify_keys(df["source_dataset"].value_counts().sort_index().to_dict()),
    }


def stringify_keys(values: dict[object, object]) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.items()}


def write_splits(method: str, splits: dict[str, pd.DataFrame], config: SplitConfig) -> None:
    output_dir = config.output_root / method
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": method,
        "inputs": [str(path) for path in config.inputs],
        "label_col": config.label_col,
        "normal_label": config.normal_label,
        "val_size": config.val_size,
        "test_size": config.test_size,
        "seed": config.seed,
        "stratify_source": config.stratify_source,
        "splits": {},
    }

    for split_name, split_df in splits.items():
        split_df.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        summary["splits"][split_name] = summarize_split(split_df, config)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {method} splits to {output_dir}")
    for split_name, split_df in splits.items():
        counts = split_df[config.label_col].value_counts().sort_index().to_dict()
        print(f"  {split_name:<5} rows={len(split_df):>6} labels={counts}")


def main() -> None:
    config = parse_args()
    df = load_inputs(config.inputs, config.label_col, config.seed)

    methods = resolve_methods(config.method)
    for method in methods:
        if method == "regular":
            splits = split_regular(df, config)
        elif method == "stratified":
            splits = split_stratified(df, config)
        elif method == "train_only_nontoxic":
            splits = split_train_only_nontoxic(df, config)
        else:
            raise ValueError(f"Unknown method: {method}")
        write_splits(method, splits, config)


def resolve_methods(method: SplitMethod) -> list[str]:
    if method == "all":
        return ["regular", "stratified", "train_only_nontoxic"]
    if method == "both":
        return ["stratified", "train_only_nontoxic"]
    if method == "normal_train_mixed_eval":
        return ["train_only_nontoxic"]
    return [method]


if __name__ == "__main__":
    main()
