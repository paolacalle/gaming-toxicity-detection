# libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer

from pathlib import Path
import pandas as pd

from src.tokenizer import tokenize

# roots
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WOT_PATH = _PROJECT_ROOT / "data/processed_data/wot/wot.parquet"
_DOTA_PATH = _PROJECT_ROOT / "data/processed_data/dota/dota.parquet"

# base tfidf config - tokenizer swappable
_TFIDF_BASE = dict(ngram_range=(1, 2), min_df=1, max_df=0.95,
                   sublinear_tf=True, norm="l2")

# scores
_scorers = {
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
}

# cv
_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)


def eval_step(label, preprocessing_impact=None, datasets=("WoT", "Dota"),
              clf=None, tokenizer_fn=tokenize, dataset_paths=None):
    """
    Evaluate a preprocessing step by training a quick baseline model
    via cross-validation on the specified datasets.

    Parameters
    ----------
    label : str
        Name of the current preprocessing step.
    preprocessing_impact : pd.DataFrame or None
        Accumulator DataFrame from previous eval_step calls.
    datasets : tuple
        Dataset names to evaluate. Ignored when ``dataset_paths`` is provided,
        in which case the keys of ``dataset_paths`` are used directly.
    clf : sklearn estimator or None
        Classifier to use. Defaults to balanced LogisticRegression.
    tokenizer_fn : callable or None
        Tokenizer for TF-IDF. ``None`` uses the built-in analyzer.
    dataset_paths : dict or None
        Mapping of {dataset_name: path_to_parquet}. When provided it overrides
        the hard-coded WoT/Dota paths, making the function usable with any
        dataset.

    Returns
    -------
    pd.DataFrame
        Updated accumulator with the new step's scores.
    """
    if clf is None:
        clf = LogisticRegression(max_iter=1000, random_state=7524, class_weight="balanced")

    if tokenizer_fn is None:
        tfidf_cfg = _TFIDF_BASE.copy()
    else:
        tfidf_cfg = {**_TFIDF_BASE, "analyzer": "word",
                     "tokenizer": tokenizer_fn, "token_pattern": None}

    if dataset_paths is None:
        dataset_paths = {"WoT": _WOT_PATH, "Dota": _DOTA_PATH}

    rows = []
    for name, path in dataset_paths.items():
        if datasets is not None and name not in datasets:
            continue
        df = pd.read_parquet(path)[["message", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        X, y = df["message"].values, df["label"].values

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_cfg)),
            ("clf", clf)
        ])

        cv_res = cross_validate(pipe, X, y, cv=_cv, scoring=_scorers, n_jobs=-1)

        rows.append({
            "step": label,
            "dataset": name,
            "rows": len(df),
            "f1_macro": round(cv_res["test_f1_macro"].mean(), 4),
            "f1_weighted": round(cv_res["test_f1_weighted"].mean(), 4),
            "recall_macro": round(cv_res["test_recall_macro"].mean(), 4),
            "precision_macro": round(cv_res["test_precision_macro"].mean(), 4),
        })

    new = pd.DataFrame(rows)

    if preprocessing_impact is not None and len(preprocessing_impact) > 0:
        for i, row in new.iterrows():
            prev = preprocessing_impact[preprocessing_impact["dataset"] == row["dataset"]]
            if len(prev):
                prev = prev.iloc[-1]
                new.at[i, "f1_macro_delta"] = round(row["f1_macro"] - prev["f1_macro"], 4)
                new.at[i, "recall_delta"] = round(row["recall_macro"] - prev["recall_macro"], 4)
                new.at[i, "precision_delta"] = round(row["precision_macro"] - prev["precision_macro"], 4)
    else:
        new["f1_macro_delta"] = 0.0
        new["recall_delta"] = 0.0
        new["precision_delta"] = 0.0

    print(new.to_string(index=False))
    return pd.concat([preprocessing_impact, new], ignore_index=True) if preprocessing_impact is not None else new
