from __future__ import annotations

import argparse
import json
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.svm import OneClassSVM

from stopwords import STOPWORDS

warnings.filterwarnings("ignore")


DEFAULT_TFIDF = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.90,
    "sublinear_tf": True,
    "norm": "l2",
}


@dataclass
class AnomalyConfig:
    train_path: Path
    tune_path: Path
    test_path: Path
    output_model: Path
    output_report: Path | None
    text_col: str
    label_col: str
    normal_label: int
    threshold_quantile: float
    seed: int
    use_custom_stopwords: bool
    tfidf: dict[str, Any]


def parse_args() -> AnomalyConfig:
    parser = argparse.ArgumentParser(
        description="Generic anomaly-detection pipeline for text: normal-only train, normal-only threshold tuning, mixed test evaluation."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--tune-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--text-col", default="clean_message")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--normal-label", type=int, default=0)
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.05,
        help="Lower decision-score quantile on normal tuning data used as the anomaly cutoff.",
    )
    parser.add_argument("--seed", type=int, default=7524)
    parser.add_argument("--use-custom-stopwords", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be between 0 and 1.")

    tfidf = dict(DEFAULT_TFIDF)
    if args.use_custom_stopwords:
        tfidf["stop_words"] = STOPWORDS

    return AnomalyConfig(
        train_path=args.train_path,
        tune_path=args.tune_path,
        test_path=args.test_path,
        output_model=args.output_model,
        output_report=args.output_report,
        text_col=args.text_col,
        label_col=args.label_col,
        normal_label=args.normal_label,
        threshold_quantile=args.threshold_quantile,
        seed=args.seed,
        use_custom_stopwords=args.use_custom_stopwords,
        tfidf=tfidf,
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot read parquet file {path}. Install a parquet engine such as `pyarrow` "
                "in the active environment, or pass a .csv file instead."
            ) from exc

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type for {path}. Expected .parquet or .csv.")


def load_frame(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = read_table(path).copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = df[label_col].astype(int)
    return df[[text_col, label_col]]


def filter_normal(df: pd.DataFrame, label_col: str, normal_label: int) -> pd.DataFrame:
    return df.loc[df[label_col] == normal_label].reset_index(drop=True)


def to_binary_anomaly_labels(y: pd.Series, normal_label: int) -> np.ndarray:
    return (y.to_numpy() != normal_label).astype(int)


def build_detectors(config: AnomalyConfig) -> dict[str, Any]:
    return {
        "IsolationForest": IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=config.seed,
            n_jobs=-1,
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=max(config.threshold_quantile, 0.01)),
        "SGDOneClassSVM": SGDOneClassSVM(
            nu=max(config.threshold_quantile, 0.01),
            random_state=config.seed,
            max_iter=2000,
            tol=1e-3,
        ),
    }


def decision_scores(model: Any, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).reshape(-1)
    if hasattr(model, "score_samples"):
        return np.asarray(model.score_samples(X)).reshape(-1)
    raise ValueError(f"Model {type(model).__name__} does not expose decision scores.")


def fit_and_score(
    detector: Any,
    X_train_vec,
    X_tune_vec,
    X_test_vec,
    threshold_quantile: float,
) -> dict[str, Any]:
    model = clone(detector)
    model.fit(X_train_vec)

    tune_scores = decision_scores(model, X_tune_vec)
    threshold = float(np.quantile(tune_scores, threshold_quantile))
    
    tune_pred = (tune_scores < threshold).astype(int)

    test_scores = decision_scores(model, X_test_vec)
    test_pred = (test_scores < threshold).astype(int)

    return {
        "model": model,
        "threshold": threshold,
        "tune_scores": tune_scores,
        "tune_pred": tune_pred,
        "test_scores": test_scores,
        "test_pred": test_pred,
    }


def summarize_detector(
    detector_name: str,
    outputs: dict[str, Any],
    y_test_binary: np.ndarray,
) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_binary,
        outputs["test_pred"],
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_test_binary, outputs["test_pred"], labels=[0, 1]).ravel()
    tune_flag_rate = float(outputs["tune_pred"].mean())

    return {
        "Detector": detector_name,
        "Threshold": round(outputs["threshold"], 6),
        "Tune Flag Rate": round(tune_flag_rate, 4),
        "Test Precision": round(float(precision), 4),
        "Test Recall": round(float(recall), 4),
        "Test F1": round(float(f1), 4),
        "Test FPR": round(float(fp / max(tn + fp, 1)), 4),
        "Test TPR": round(float(tp / max(tp + fn, 1)), 4),
    }


def save_bundle(
    vectorizer: TfidfVectorizer,
    detector_name: str,
    outputs: dict[str, Any],
    config: AnomalyConfig,
) -> None:
    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "detector_name": detector_name,
            "detector": outputs["model"],
            "threshold": outputs["threshold"],
            "normal_label": config.normal_label,
            "text_col": config.text_col,
            "label_col": config.label_col,
        },
        config.output_model,
    )


def save_report(
    config: AnomalyConfig,
    detector_results: pd.DataFrame,
    best_detector_name: str,
    best_outputs: dict[str, Any],
    y_test_binary: np.ndarray,
) -> None:
    if config.output_report is None:
        return

    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_path": str(config.train_path),
        "tune_path": str(config.tune_path),
        "test_path": str(config.test_path),
        "normal_label": config.normal_label,
        "threshold_quantile": config.threshold_quantile,
        "best_detector": best_detector_name,
        "detector_results": detector_results.to_dict(orient="records"),
        "test_classification_report": classification_report(
            y_test_binary,
            best_outputs["test_pred"],
            target_names=["normal", "anomaly"],
            zero_division=0,
            output_dict=True,
        ),
    }
    config.output_report.write_text(json.dumps(payload, indent=2))


def main() -> None:
    config = parse_args()
    np.random.seed(config.seed)

    train_df = load_frame(config.train_path, config.text_col, config.label_col)
    tune_df = load_frame(config.tune_path, config.text_col, config.label_col)
    test_df = load_frame(config.test_path, config.text_col, config.label_col)

    train_normal = filter_normal(train_df, config.label_col, config.normal_label)
    tune_normal = filter_normal(tune_df, config.label_col, config.normal_label)

    if train_normal.empty:
        raise ValueError("Training split has no normal examples after filtering.")
    if tune_normal.empty:
        raise ValueError("Tuning split has no normal examples after filtering.")

    y_test_binary = to_binary_anomaly_labels(test_df[config.label_col], config.normal_label)

    print(f"Normal train rows: {len(train_normal):,}")
    print(f"Normal tune rows:  {len(tune_normal):,}")
    print(f"Mixed test rows:   {len(test_df):,}")
    print(f"Mixed test anomaly rate: {y_test_binary.mean():.4f}")

    vectorizer = TfidfVectorizer(**config.tfidf)
    X_train_vec = vectorizer.fit_transform(train_normal[config.text_col])
    X_tune_vec = vectorizer.transform(tune_normal[config.text_col])
    X_test_vec = vectorizer.transform(test_df[config.text_col])

    rows: list[dict[str, Any]] = []
    all_outputs: dict[str, dict[str, Any]] = {}
    
    models = build_detectors(config)

    for detector_name, detector in models.items():
        outputs = fit_and_score(
            detector=detector,
            X_train_vec=X_train_vec,
            X_tune_vec=X_tune_vec,
            X_test_vec=X_test_vec,
            threshold_quantile=config.threshold_quantile,
        )
        all_outputs[detector_name] = outputs
        row = summarize_detector(detector_name, outputs, y_test_binary)
        rows.append(row)
        print(
            f"{detector_name:<16} "
            f"tune_flag_rate={row['Tune Flag Rate']:.4f} "
            f"test_f1={row['Test F1']:.4f} "
            f"test_recall={row['Test Recall']:.4f}"
        )

    detector_results = pd.DataFrame(rows).sort_values(["Test F1", "Test Recall"], ascending=False).reset_index(drop=True)
    best_detector_name = str(detector_results.iloc[0]["Detector"])
    best_outputs = all_outputs[best_detector_name]

    print("\nDetector ranking:")
    print(detector_results.to_string(index=False))
    print("\nBest detector holdout report:\n")
    print(
        classification_report(
            y_test_binary,
            best_outputs["test_pred"],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )

    save_bundle(vectorizer, best_detector_name, best_outputs, config)
    save_report(config, detector_results, best_detector_name, best_outputs, y_test_binary)
    print(f"Saved model bundle to {config.output_model}")
    if config.output_report is not None:
        print(f"Saved report to {config.output_report}")
        
    return best_detector_name, best_outputs, detector_results

if __name__ == "__main__":
    main()
