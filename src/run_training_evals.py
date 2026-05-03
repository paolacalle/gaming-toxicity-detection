#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, get_model_config, list_model_configs, list_models
from src.models.supervised import SupervisedTextModel

warnings.filterwarnings("ignore")

DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "data" / "results" / "training_evaluations.csv"
DEFAULT_SEED = 7524
CANONICAL_METHODS = ("regular", "stratified", "train_only_nontoxic")
MODEL_ALIASES = {
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "LinearSVC": "linear_svc",
    "XGBoost": "xgboost",
    "SGDOneClassSVM": "sgd_one_class_svm",
    "IsolationForest": "isolation_forest",
}


@dataclass(frozen=True)
class EvalConfig:
    split_root: Path
    output_csv: Path
    methods: list[str]
    models: list[str]
    model_configs: list[str]
    text_col: str
    label_col: str
    normal_label: int
    binary: bool
    oversample: bool
    seed: int


@dataclass(frozen=True)
class ModelRun:
    model_name: str
    config_name: str
    estimator: BaseEstimator
    training_mode: str

    @property
    def display_name(self) -> str:
        return f"{self.model_name}:{self.config_name}"

def read_split(path: Path, config: EvalConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_parquet(path)
    for col in (config.text_col, config.label_col):
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column {col!r}. Available: {list(df.columns)}")
    df = df.copy()
    df[config.text_col] = df[config.text_col].fillna("").astype(str)
    df[config.label_col] = df[config.label_col].astype(int)
    if config.binary:
        df[config.label_col] = (df[config.label_col] != config.normal_label).astype(int)
    return df


def load_holdout(method_dir: Path, config: EvalConfig) -> dict[str, pd.DataFrame]:
    return {
        "train": read_split(method_dir / "train.parquet", config),
        "val": read_split(method_dir / "val.parquet", config),
        "test": read_split(method_dir / "test.parquet", config),
    }


def load_folds(method_dir: Path, config: EvalConfig) -> list[tuple[str, dict[str, pd.DataFrame]]]:
    folds_dir = method_dir / "folds"
    if not folds_dir.exists():
        return []

    folds = []
    for fold_dir in sorted(folds_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        folds.append(
            (
                fold_dir.name,
                {
                    "train": read_split(fold_dir / "train.parquet", config),
                    "val": read_split(fold_dir / "val.parquet", config),
                },
            )
        )
    return folds


def metric_average(y_true: pd.Series) -> str:
    return "binary" if y_true.nunique() == 2 else "macro"


def score_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    normal_label: int = 0,
) -> dict[str, object]:
    average = metric_average(y_true)
    pos_label = 1 if average == "binary" else None
    kwargs = {"average": average, "zero_division": 0}
    if pos_label is not None:
        kwargs["pos_label"] = pos_label

    row: dict[str, object] = {
        "rows": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, **kwargs)), 4),
        "recall": round(float(recall_score(y_true, y_pred, **kwargs)), 4),
        "precision": round(float(precision_score(y_true, y_pred, **kwargs)), 4),
        **confusion_rates(y_true, y_pred),
        "label_counts": json.dumps(stringify_counts(y_true)),
        "prediction_counts": json.dumps(stringify_counts(pd.Series(y_pred))),
    }

    auc = compute_auc(y_true, scores, normal_label)
    if auc is not None:
        row["auc"] = auc
    return row


def confusion_rates(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(set(y_true).union(set(y_pred)))
    if len(labels) == 2:
        negative, positive = labels
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[negative, positive]).ravel()
        return format_rates(tn, fp, fn, tp)

    rates = []
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)
    for label in labels:
        binary_true = (y_true_series == label).astype(int)
        binary_pred = (y_pred_series == label).astype(int)
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()
        rates.append(format_rates(tn, fp, fn, tp))

    return {
        metric: round(float(np.mean([rate[metric] for rate in rates])), 4)
        for metric in ("fpr", "fnr", "tpr", "tnr")
    }


def format_rates(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    return {
        "fpr": round(float(fp / max(fp + tn, 1)), 4),
        "fnr": round(float(fn / max(fn + tp, 1)), 4),
        "tpr": round(float(tp / max(tp + fn, 1)), 4),
        "tnr": round(float(tn / max(tn + fp, 1)), 4),
    }


def compute_auc(y_true: pd.Series, scores: np.ndarray | None, normal_label: int) -> float | None:
    if scores is None:
        return None
    try:
        classes = sorted(y_true.unique())
        if len(classes) == 2:
            positive = (y_true != normal_label).astype(int)
            return round(float(roc_auc_score(positive, scores)), 4)
        return round(float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro")), 4)
    except Exception:
        return None


def stringify_counts(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values.value_counts().sort_index().to_dict().items()}


def supervised_scores(
    model_run: ModelRun,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: EvalConfig,
) -> dict[str, object]:
    if train_df[config.label_col].nunique() < 2:
        raise ValueError("Supervised classifiers require at least two training classes.")

    started = time.time()
    model = clone(model_run.estimator)
    model.fit(train_df[config.text_col], train_df[config.label_col])
    y_pred = model.predict(eval_df[config.text_col])
    scores = decision_scores(model, eval_df[config.text_col])
    row = score_predictions(eval_df[config.label_col], y_pred, scores, config.normal_label)
    row.update(
        {
            "model": model_run.model_name,
            "model_config": model_run.config_name,
            "model_experiment": model_run.display_name,
            "training_mode": model_run.training_mode,
            "fit_seconds": round(time.time() - started, 2),
        }
    )
    return row


def decision_scores(model: BaseEstimator, X: pd.Series) -> np.ndarray | None:
    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim == 2 and scores.shape[1] == 2:
                return scores[:, 1]
            return scores
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
    except Exception:
        return None
    return None


def anomaly_scores(
    model_run: ModelRun,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: EvalConfig,
) -> dict[str, object]:
    normal_train = train_df.loc[train_df[config.label_col] == config.normal_label]
    if normal_train.empty:
        raise ValueError("Anomaly model needs at least one normal training row.")

    started = time.time()
    model = clone(model_run.estimator)
    model.fit(normal_train[config.text_col])
    raw_scores = decision_scores(model, eval_df[config.text_col])
    if raw_scores is None:
        raw_scores = np.zeros(len(eval_df))
    raw_pred = model.predict(eval_df[config.text_col])
    y_true_binary = (eval_df[config.label_col] != config.normal_label).astype(int)
    y_pred_binary = (np.asarray(raw_pred) == -1).astype(int)
    anomaly_scores_for_auc = -np.asarray(raw_scores)

    row = score_predictions(y_true_binary, y_pred_binary, anomaly_scores_for_auc, normal_label=0)
    row.update(
        {
            "model": model_run.model_name,
            "model_config": model_run.config_name,
            "model_experiment": model_run.display_name,
            "training_mode": model_run.training_mode,
            "fit_seconds": round(time.time() - started, 2),
        }
    )
    return row


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def selected_config_names(model_name: str, config: EvalConfig) -> list[str]:
    available = list_model_configs(model_name)
    if "all" in config.model_configs:
        return list(available)
    return [name for name in config.model_configs if name in available]


def build_model_runs(config: EvalConfig, task_type: str) -> list[ModelRun]:
    requested = [normalize_model_name(name) for name in config.models]
    if "all" in requested:
        requested = list_models(task_type)

    runs = []
    for model_name in requested:
        if model_name not in list_models(task_type):
            continue
        model_class = build_model(model_name).__class__
        if task_type == "supervised" and not issubclass(model_class, SupervisedTextModel):
            continue
        for config_name in selected_config_names(model_name, config):
            kwargs = {"seed": config.seed}
            if task_type == "supervised" and config.oversample:
                kwargs["oversample"] = True
            model = build_model(model_name, config_name, **kwargs)
            estimator = model.build()
            if not isinstance(estimator, BaseEstimator):
                continue
            get_model_config(model_name, config_name)
            runs.append(
                ModelRun(
                    model_name=model_name,
                    config_name=config_name,
                    estimator=estimator,
                    training_mode=task_type,
                )
            )
    return runs


def evaluate_holdout(method: str, splits: dict[str, pd.DataFrame], config: EvalConfig) -> list[dict[str, object]]:
    rows = []
    if method == "train_only_nontoxic":
        for model_run in build_model_runs(config, "unsupervised"):
            for split_name in ("val", "test"):
                row = anomaly_scores(model_run, splits["train"], splits[split_name], config)
                row.update({"split_method": method, "evaluation": split_name, "fold": ""})
                rows.append(row)
        return rows

    for model_run in build_model_runs(config, "supervised"):
        for split_name in ("val", "test"):
            row = supervised_scores(model_run, splits["train"], splits[split_name], config)
            row.update({"split_method": method, "evaluation": split_name, "fold": ""})
            rows.append(row)
    return rows


def evaluate_folds(method: str, folds: Iterable[tuple[str, dict[str, pd.DataFrame]]], config: EvalConfig) -> list[dict[str, object]]:
    rows = []
    for fold_name, split in folds:
        if method == "train_only_nontoxic":
            for model_run in build_model_runs(config, "unsupervised"):
                row = anomaly_scores(model_run, split["train"], split["val"], config)
                row.update({"split_method": method, "evaluation": "fold_val", "fold": fold_name})
                rows.append(row)
            continue

        for model_run in build_model_runs(config, "supervised"):
            row = supervised_scores(model_run, split["train"], split["val"], config)
            row.update({"split_method": method, "evaluation": "fold_val", "fold": fold_name})
            rows.append(row)
    return rows


def summarize_fold_means(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame([row for row in rows if row.get("evaluation") == "fold_val"])
    if df.empty:
        return []

    metric_cols = ["accuracy", "f1", "recall", "precision", "fpr", "fnr", "tpr", "tnr", "auc", "fit_seconds"]
    present_metrics = [col for col in metric_cols if col in df.columns]
    group_cols = ["split_method", "model", "model_config", "model_experiment", "training_mode"]
    summaries = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update({"evaluation": "fold_mean", "fold": "", "rows": int(group["rows"].sum())})
        for col in present_metrics:
            row[col] = round(float(group[col].dropna().mean()), 4) if group[col].notna().any() else np.nan
            row[f"{col}_std"] = round(float(group[col].dropna().std(ddof=0)), 4) if group[col].notna().any() else np.nan
        summaries.append(row)
    return summaries


def add_experiment_names(rows: list[dict[str, object]]) -> None:
    for row in rows:
        parts = [
            str(row.get("split_method", "")),
            str(row.get("evaluation", "")),
            str(row.get("model_experiment", row.get("model", ""))),
        ]
        fold = row.get("fold")
        if fold:
            parts.append(str(fold))
        row["experiment_name"] = "__".join(part for part in parts if part)


def run(config: EvalConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in config.methods:
        method_dir = config.split_root / method
        
        if not method_dir.exists():
            print(f"Skipping {method}: {method_dir} does not exist")
            continue

        print(f"Evaluating {method}")
        holdout = load_holdout(method_dir, config)
        method_rows = evaluate_holdout(method, holdout, config)
        method_rows.extend(evaluate_folds(method, load_folds(method_dir, config), config))
        method_rows.extend(summarize_fold_means(method_rows))
        add_experiment_names(method_rows)
        rows.extend(method_rows)

    if not rows:
        raise ValueError(f"No evaluations were run from split root {config.split_root}")

    results = pd.DataFrame(rows)
    ordered = [
        "experiment_name",
        "split_method",
        "evaluation",
        "fold",
        "model",
        "model_config",
        "model_experiment",
        "training_mode",
        "rows",
        "accuracy",
        "f1",
        "recall",
        "precision",
        "fpr",
        "fnr",
        "tpr",
        "tnr",
        "auc",
        "fit_seconds",
        "accuracy_std",
        "f1_std",
        "recall_std",
        "precision_std",
        "fpr_std",
        "fnr_std",
        "tpr_std",
        "tnr_std",
        "auc_std",
        "fit_seconds_std",
        "label_counts",
        "prediction_counts",
    ]
    return results[[col for col in ordered if col in results.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training evaluations on data splits.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT, help="Root directory containing split subdirectories.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_RESULTS_CSV, help="Path to save the evaluation results CSV.")
    parser.add_argument("--methods", nargs="+", default=list(CANONICAL_METHODS), help=f"Split methods to evaluate. Available: {', '.join(CANONICAL_METHODS)}")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model names to evaluate. Use registry names or legacy display names. "
            "Use 'all' to run all compatible registry models for each split method."
        ),
    )
    parser.add_argument(
        "--model-configs",
        nargs="+",
        default=["all"],
        help="MODEL_CONFIGS variants to run for each model. Use 'all' for every registered config.",
    )
    parser.add_argument("--text-col", type=str, default="message", help="Name of the text column in the splits.")
    parser.add_argument("--label-col", type=str, default="label", help="Name of the label column in the splits.")
    parser.add_argument("--normal-label", type=int, default=0, help="Value of the normal (non-toxic) label.")
    parser.add_argument("--binary", action="store_true", help="Whether to treat labels as binary (toxic vs non-toxic).")
    parser.add_argument("--oversample", action="store_true", help="Whether to apply oversampling to the training data for supervised models.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")

    args = parser.parse_args()
    config = EvalConfig(
        split_root=args.split_root,
        output_csv=args.output_csv,
        methods=args.methods,
        models=args.models,
        model_configs=args.model_configs,
        text_col=args.text_col,
        label_col=args.label_col,
        normal_label=args.normal_label,
        binary=args.binary,
        oversample=args.oversample,
        seed=args.seed,
    )
    results = run(config)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_csv, index=False)
    print(f"Saved evaluation results to {config.output_csv}")


if __name__ == "__main__":
    main()
