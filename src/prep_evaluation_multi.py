from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

import pandas as pd

from src.tokenizer import tokenize

# base TF-IDF config — tokenizer is swappable per call
_TFIDF_BASE = dict(
    ngram_range=(1, 2), min_df=1, max_df=0.95,
    sublinear_tf=True, norm="l2"
)

# scoring metrics for cross-validation
_SCORERS = {
    "f1_macro":        make_scorer(f1_score,        average="macro",    zero_division=0),
    "f1_weighted":     make_scorer(f1_score,        average="weighted", zero_division=0),
    "recall_macro":    make_scorer(recall_score,    average="macro",    zero_division=0),
    "precision_macro": make_scorer(precision_score, average="macro",    zero_division=0),
}

# reproducibility
SEED = 7524
_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def eval_step(label, dataset_paths, preprocessing_impact=None,
              clf=None, tokenizer_fn=tokenize):
    """
    Evaluate a preprocessing step via stratified 5-fold cross-validation.

    Parameters
    ----------
    label : str
        Name of the current preprocessing step.
    dataset_paths : dict
        {dataset_name: path_to_parquet} — one entry per dataset to evaluate.
    preprocessing_impact : pd.DataFrame or None
        Accumulator DataFrame from previous eval_step calls.
    clf : sklearn estimator or None
        Classifier to use. Defaults to balanced LogisticRegression.
    tokenizer_fn : callable or None
        Tokenizer for TF-IDF. None uses the built-in analyzer.

    Returns
    -------
    pd.DataFrame
        Updated accumulator with the new step's scores.
    """
    if clf is None:
        clf = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")

    # attach custom tokenizer if provided
    if tokenizer_fn is None:
        tfidf_cfg = _TFIDF_BASE.copy()
    else:
        tfidf_cfg = {**_TFIDF_BASE, "analyzer": "word",
                     "tokenizer": tokenizer_fn, "token_pattern": None}

    rows = []
    for name, path in dataset_paths.items():
        # load dataset
        df = pd.read_parquet(path)[["message", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        X, y = df["message"].values, df["label"].values

        # pipeline clones clf each fold automatically via cross_validate
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_cfg)),
            ("clf", clone(clf)),
        ])

        cv_res = cross_validate(pipe, X, y, cv=_CV, scoring=_SCORERS, n_jobs=-1)

        rows.append({
            "step":            label,
            "dataset":         name,
            "rows":            len(df),
            "f1_macro":        round(cv_res["test_f1_macro"].mean(),        4),
            "f1_weighted":     round(cv_res["test_f1_weighted"].mean(),     4),
            "recall_macro":    round(cv_res["test_recall_macro"].mean(),    4),
            "precision_macro": round(cv_res["test_precision_macro"].mean(), 4),
        })

    new = pd.DataFrame(rows)

    # compute deltas vs the previous step for the same dataset
    if preprocessing_impact is not None and len(preprocessing_impact) > 0:
        for i, row in new.iterrows():
            prev = preprocessing_impact[preprocessing_impact["dataset"] == row["dataset"]]
            if len(prev):
                prev = prev.iloc[-1]
                new.at[i, "f1_macro_delta"]  = round(row["f1_macro"]        - prev["f1_macro"],        4)
                new.at[i, "recall_delta"]    = round(row["recall_macro"]    - prev["recall_macro"],    4)
                new.at[i, "precision_delta"] = round(row["precision_macro"] - prev["precision_macro"], 4)
    else:
        new["f1_macro_delta"]  = 0.0
        new["recall_delta"]    = 0.0
        new["precision_delta"] = 0.0

    print(new.to_string(index=False))
    return pd.concat([preprocessing_impact, new], ignore_index=True) if preprocessing_impact is not None else new
