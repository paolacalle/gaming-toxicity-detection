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
_WOT_PATH     = _PROJECT_ROOT / "data/processed_data/wot/wot.parquet"
_DOTA_PATH    = _PROJECT_ROOT / "data/processed_data/dota/dota.parquet"

# tfidf pipeline
_pipe_cfg = dict(ngram_range=(1, 2), min_df=1, max_df=0.95,
                 sublinear_tf=True, norm="l2",
                 analyzer="word", tokenizer=tokenize, token_pattern=None)

# scores 
_scorers = {
    "f1_macro":        make_scorer(f1_score,        average="macro",    zero_division=0),
    "f1_weighted":     make_scorer(f1_score,         average="weighted", zero_division=0),
    "recall_macro":    make_scorer(recall_score,     average="macro",    zero_division=0),
    "precision_macro": make_scorer(precision_score,  average="macro",    zero_division=0),
}

# cv
_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)

# evaluation
def eval_step(label, preprocessing_impact=None, datasets=("WoT", "Dota"), clf=None):
    if clf is None:
        clf = LogisticRegression(max_iter=1000, random_state=7524, class_weight="balanced")

    rows = []
    for name, path in [("WoT", _WOT_PATH), ("Dota", _DOTA_PATH)]:
        if name not in datasets:
            continue
        df = pd.read_parquet(path)[["message", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        X, y = df["message"].values, df["label"].values

        # tf-idf pipeline
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**_pipe_cfg)),
            ("clf",   clf)
        ])

        # cross-validation
        cv_res = cross_validate(pipe, X, y, cv=_cv, scoring=_scorers, n_jobs=-1)

        # append results
        rows.append({
            "step":            label,
            "dataset":         name,
            "rows":            len(df),
            "f1_macro":        round(cv_res["test_f1_macro"].mean(),       4),
            "f1_weighted":     round(cv_res["test_f1_weighted"].mean(),    4),
            "recall_macro":    round(cv_res["test_recall_macro"].mean(),   4),
            "precision_macro": round(cv_res["test_precision_macro"].mean(), 4),
        })

    # as dataframe
    new = pd.DataFrame(rows)

    # compute deltas vs previous step for same dataset
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

    # print results
    print(new.to_string(index=False))
    return pd.concat([preprocessing_impact, new], ignore_index=True) if preprocessing_impact is not None else new

    # as dataframe
    new = pd.DataFrame(rows)

    # compute deltas vs previous step for same dataset
    if preprocessing_impact is not None and len(preprocessing_impact) > 0:
        for i, row in new.iterrows():
            prev = preprocessing_impact[preprocessing_impact["dataset"] == row["dataset"]]
            if len(prev):
                prev = prev.iloc[-1]
                new.at[i, "f1_macro_delta"]      = round(row["f1_macro"]       - prev["f1_macro"],       4)
                new.at[i, "recall_delta"]        = round(row["recall_macro"]   - prev["recall_macro"],   4)
                new.at[i, "precision_delta"]     = round(row["precision_macro"] - prev["precision_macro"], 4)
    else:
        new["f1_macro_delta"]  = 0.0
        new["recall_delta"]    = 0.0
        new["precision_delta"] = 0.0

    # print results 
    print(new.to_string(index=False))
    return pd.concat([preprocessing_impact, new], ignore_index=True) if preprocessing_impact is not None else new
