from __future__ import annotations

import argparse
import json
import time
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from stopwords import STOPWORDS

warnings.filterwarnings("ignore")

try:
    import optuna
    from optuna.samplers import TPESampler
except ModuleNotFoundError:
    optuna = None
    TPESampler = None

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_TFIDF = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.95,
    "sublinear_tf": True,
    "norm": "l2",
}


@dataclass
class PipelineConfig:
    train_path: Path
    test_path: Path
    output_model: Path
    output_report: Path | None
    text_col: str
    label_col: str
    seed: int
    cv_folds: int
    scoring: str
    optuna_trials: int
    binary_threshold: int | None
    use_custom_stopwords: bool
    tfidf: dict[str, Any]


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Generic text classification pipeline with TF-IDF, oversampling, CV model selection, and final holdout evaluation."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--text-col", default="clean_message")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--seed", type=int, default=7524)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--scoring", default="f1_macro")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument(
        "--binary-threshold",
        type=int,
        default=None,
        help="If set, convert labels to binary using label > threshold.",
    )
    parser.add_argument(
        "--use-custom-stopwords",
        action="store_true",
        help="Add src/stopwords.py to the TF-IDF vectorizer.",
    )
    args = parser.parse_args()

    tfidf = dict(DEFAULT_TFIDF)
    if args.use_custom_stopwords:
        tfidf["stop_words"] = STOPWORDS

    return PipelineConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        output_model=args.output_model,
        output_report=args.output_report,
        text_col=args.text_col,
        label_col=args.label_col,
        seed=args.seed,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        optuna_trials=args.optuna_trials,
        binary_threshold=args.binary_threshold,
        use_custom_stopwords=args.use_custom_stopwords,
        tfidf=tfidf,
    )


def make_cv(config: PipelineConfig) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.seed,
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


def load_split(path: Path, text_col: str, label_col: str, binary_threshold: int | None) -> tuple[pd.Series, pd.Series]:
    df = read_table(path)
    X = df[text_col].fillna("").astype(str)
    y = df[label_col].astype(int)
    if binary_threshold is not None:
        y = (y > binary_threshold).astype(int)
    return X, y


def make_oversamplers(seed: int) -> dict[str, BaseEstimator]:
    return {
        "RandomOverSampler": RandomOverSampler(random_state=seed),
        "SMOTE": SMOTE(random_state=seed),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=seed),
        "ADASYN": ADASYN(random_state=seed),
    }


def make_reference_pipeline(config: PipelineConfig, oversampler: BaseEstimator) -> ImbPipeline:
    return ImbPipeline(
        [
            ("tfidf", TfidfVectorizer(**config.tfidf)),
            ("oversample", oversampler),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=config.seed, n_jobs=1)),
        ]
    )


def evaluate_oversamplers(
    X_train: pd.Series,
    y_train: pd.Series,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    cv = make_cv(config)

    for name, sampler in make_oversamplers(config.seed).items():
        started = time.time()
        results = cross_validate(
            make_reference_pipeline(config, sampler),
            X_train,
            y_train,
            cv=cv,
            scoring=["f1_macro", "f1_weighted", "accuracy"],
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "Oversampler": name,
                "CV Macro F1": round(results["test_f1_macro"].mean(), 4),
                "CV Weighted F1": round(results["test_f1_weighted"].mean(), 4),
                "Accuracy": round(results["test_accuracy"].mean(), 4),
                "Std": round(results["test_f1_macro"].std(), 4),
                "Time (s)": round(time.time() - started, 1),
            }
        )
        print(f"{name:<20} macro_f1={results['test_f1_macro'].mean():.4f} +/- {results['test_f1_macro'].std():.4f}")

    comparison = pd.DataFrame(rows).sort_values("CV Macro F1", ascending=False).reset_index(drop=True)
    return comparison, str(comparison.iloc[0]["Oversampler"])


def start_study_optuna(objective, n_trials: int, seed: int) -> optuna.Study:
    if optuna is None or TPESampler is None:
        raise ModuleNotFoundError(
            "optuna is required for hyperparameter tuning. Install dependencies with `pip install -r requirements.txt`."
        )
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def build_model_candidates(config: PipelineConfig, oversampler_name: str) -> dict[str, ImbPipeline]:
    oversampler = make_oversamplers(config.seed)[oversampler_name]
    cv = make_cv(config)

    return {
        "Logistic Regression": ImbPipeline(
            [
                ("tfidf", TfidfVectorizer(**config.tfidf)),
                ("oversample", clone(oversampler)),
                (
                    "clf",
                    LogisticRegressionCV(
                        Cs=30,
                        cv=cv,
                        scoring=config.scoring,
                        max_iter=1000,
                        random_state=config.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Naive Bayes": ImbPipeline(
            [
                ("tfidf", TfidfVectorizer(**config.tfidf)),
                ("oversample", clone(oversampler)),
                ("clf", MultinomialNB()),
            ]
        ),
        "LinearSVC": ImbPipeline(
            [
                ("tfidf", TfidfVectorizer(**config.tfidf)),
                ("oversample", clone(oversampler)),
                ("clf", LinearSVC(random_state=config.seed, max_iter=3000)),
            ]
        ),
    }


def tune_pipeline(
    name: str,
    pipeline: ImbPipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    config: PipelineConfig,
) -> ImbPipeline:
    cv = make_cv(config)

    if name == "Naive Bayes":
        def objective(trial):
            candidate = clone(pipeline)
            candidate.set_params(clf__alpha=trial.suggest_float("clf__alpha", 0.001, 2.0, log=True))
            return cross_val_score(candidate, X_train, y_train, cv=cv, scoring=config.scoring, n_jobs=1).mean()

        study = start_study_optuna(objective, config.optuna_trials, config.seed)
        pipeline.set_params(**study.best_params)
        print(f"Best params for {name}: {study.best_params}")

    if name == "LinearSVC":
        def objective(trial):
            candidate = clone(pipeline)
            candidate.set_params(
                clf__C=trial.suggest_float("clf__C", 1e-2, 10.0, log=True),
                clf__tol=trial.suggest_float("clf__tol", 1e-5, 1e-2, log=True),
            )
            return cross_val_score(candidate, X_train, y_train, cv=cv, scoring=config.scoring, n_jobs=1).mean()

        study = start_study_optuna(objective, config.optuna_trials, config.seed)
        pipeline.set_params(**study.best_params)
        print(f"Best params for {name}: {study.best_params}")

    return pipeline


def compare_models(
    X_train: pd.Series,
    y_train: pd.Series,
    config: PipelineConfig,
    oversampler_name: str,
) -> tuple[pd.DataFrame, str, ImbPipeline]:
    rows: list[dict[str, Any]] = []
    models = build_model_candidates(config, oversampler_name)
    cv = make_cv(config)

    for name, pipeline in models.items():
        tuned_pipeline = tune_pipeline(name, pipeline, X_train, y_train, config)
        started = time.time()
        results = cross_validate(
            tuned_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["f1_macro", "f1_weighted", "accuracy"],
            n_jobs=-1,
        )
        rows.append(
            {
                "Model": name,
                "CV Macro F1": round(results["test_f1_macro"].mean(), 4),
                "CV Weighted F1": round(results["test_f1_weighted"].mean(), 4),
                "Accuracy": round(results["test_accuracy"].mean(), 4),
                "Std": round(results["test_f1_macro"].std(), 4),
                "Time (s)": round(time.time() - started, 1),
            }
        )
        models[name] = tuned_pipeline
        print(f"{name:<20} macro_f1={results['test_f1_macro'].mean():.4f} +/- {results['test_f1_macro'].std():.4f}")

    comparison = pd.DataFrame(rows).sort_values("CV Macro F1", ascending=False).reset_index(drop=True)
    best_model_name = str(comparison.iloc[0]["Model"])
    return comparison, best_model_name, models[best_model_name]


def fit_evaluate_and_save(
    best_pipeline: ImbPipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    config: PipelineConfig,
) -> dict[str, Any]:
    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, config.output_model)

    print("\nHoldout classification report:\n")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model to {config.output_model}")

    return report


def save_report(
    config: PipelineConfig,
    oversampler_results: pd.DataFrame,
    best_oversampler: str,
    model_results: pd.DataFrame,
    best_model_name: str,
    holdout_report: dict[str, Any],
) -> None:
    if config.output_report is None:
        return

    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_path": str(config.train_path),
        "test_path": str(config.test_path),
        "text_col": config.text_col,
        "label_col": config.label_col,
        "binary_threshold": config.binary_threshold,
        "seed": config.seed,
        "cv_folds": config.cv_folds,
        "scoring": config.scoring,
        "tfidf": config.tfidf,
        "best_oversampler": best_oversampler,
        "best_model": best_model_name,
        "oversampler_results": oversampler_results.to_dict(orient="records"),
        "model_results": model_results.to_dict(orient="records"),
        "holdout_report": holdout_report,
    }
    config.output_report.write_text(json.dumps(payload, indent=2))
    print(f"Saved report to {config.output_report}")


def main() -> None:
    config = parse_args()
    np.random.seed(config.seed)

    X_train, y_train = load_split(
        config.train_path,
        config.text_col,
        config.label_col,
        config.binary_threshold,
    )
    X_test, y_test = load_split(
        config.test_path,
        config.text_col,
        config.label_col,
        config.binary_threshold,
    )

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")
    print(f"Labels:     {sorted(pd.Series(y_train).unique().tolist())}")

    print("\nPhase 1: oversampler comparison")
    oversampler_results, best_oversampler = evaluate_oversamplers(X_train, y_train, config)
    print("\nOversampler ranking:")
    print(oversampler_results.to_string(index=False))

    print("\nPhase 2: model comparison")
    model_results, best_model_name, best_pipeline = compare_models(
        X_train,
        y_train,
        config,
        best_oversampler,
    )
    print("\nModel ranking:")
    print(model_results.to_string(index=False))

    print("\nPhase 3: final fit and holdout evaluation")
    holdout_report = fit_evaluate_and_save(best_pipeline, X_train, y_train, X_test, y_test, config)
    save_report(
        config,
        oversampler_results,
        best_oversampler,
        model_results,
        best_model_name,
        holdout_report,
    )


if __name__ == "__main__":
    main()
