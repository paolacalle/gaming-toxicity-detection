from __future__ import annotations

from typing import Any

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone

from src.models.features import build_default_tfidf
from src.models.registry import build_model, get_model_class, list_models
from src.models.supervised import SupervisedTextModel


DEFAULT_SEED = 7524
DEFAULT_TFIDF = build_default_tfidf()

MODEL_ALIASES = {
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "LinearSVC": "linear_svc",
    "XGBoost": "xgboost",
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "linear_svc": "LinearSVC",
    "xgboost": "XGBoost",
}


def default_oversampler(seed: int = DEFAULT_SEED) -> RandomOverSampler:
    return RandomOverSampler(random_state=seed)


def build_pipe(clf: BaseEstimator, oversampler=None, tfidf=None) -> ImbPipeline:
    steps = [("tfidf", clone(tfidf if tfidf is not None else DEFAULT_TFIDF))]
    if oversampler is not None:
        steps.append(("oversample", oversampler))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def build_registered_model(
    model_name: str,
    config_name: str = "default",
    seed: int = DEFAULT_SEED,
    **overrides: Any,
) -> Any:
    return build_model(
        normalize_model_name(model_name),
        config_name=config_name,
        seed=seed,
        **overrides,
    ).build()


def default_classifiers(seed: int = DEFAULT_SEED) -> dict[str, BaseEstimator]:
    classifiers: dict[str, BaseEstimator] = {}
    for model_name in list_models("supervised"):
        model_class = get_model_class(model_name)
        if not issubclass(model_class, SupervisedTextModel):
            continue
        model = build_model(model_name, seed=seed)
        classifiers[DISPLAY_NAMES.get(model_name, model_name)] = model.build_classifier()
    return classifiers


def registered_pipelines(
    model_names: list[str] | None = None,
    config_name: str = "default",
    seed: int = DEFAULT_SEED,
    oversample: bool = False,
) -> dict[str, Any]:
    names = model_names or [
        name
        for name in list_models("supervised")
        if issubclass(get_model_class(name), SupervisedTextModel)
    ]
    return {
        DISPLAY_NAMES.get(normalize_model_name(name), normalize_model_name(name)): build_registered_model(
            name,
            config_name=config_name,
            seed=seed,
            oversample=oversample,
        )
        for name in names
    }
