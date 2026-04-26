import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.scoring import cv_score, holdout_score, ood_score, append_registry

# small synthetic binary corpus for fast tests
TEXTS = [
    "kill yourself idiot",
    "nice game well played",
    "you are trash noob",
    "good job team push",
    "report this player toxic",
    "gg wp",
] * 10
LABELS = [1, 0, 1, 0, 1, 0] * 10


def make_pipe():
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(min_df=1)),
        ('clf', LogisticRegression(max_iter=200, random_state=0)),
    ])


def test_cv_score_returns_required_keys():
    from sklearn.model_selection import StratifiedKFold
    result = cv_score(make_pipe(), pd.Series(TEXTS), pd.Series(LABELS),
                      cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0))
    assert 'cv_f2' in result
    assert 'cv_f2_std' in result
    assert 'cv_recall_macro' in result
    assert 'cv_precision_macro' in result


def test_holdout_score_returns_required_keys():
    result = holdout_score(make_pipe(), pd.Series(TEXTS), pd.Series(LABELS),
                           pd.Series(TEXTS), pd.Series(LABELS))
    assert 'test_f2' in result
    assert 'test_recall_macro' in result
    assert 'test_precision_macro' in result
    assert 'test_auc' in result
    assert 'per_class_recall' in result
    # per_class_recall must be valid JSON
    json.loads(result['per_class_recall'])


def test_ood_score_returns_required_keys():
    pipe = make_pipe()
    pipe.fit(pd.Series(TEXTS), pd.Series(LABELS))
    result = ood_score(pipe, pd.Series(TEXTS), pd.Series(LABELS))
    assert 'ood_f2' in result
    assert 'ood_recall_macro' in result
    assert 'ood_precision_macro' in result
    assert 'ood_auc' in result


def test_append_registry_creates_file(tmp_path):
    reg = tmp_path / "registry.csv"
    append_registry({'experiment': 'test', 'model': 'LR', 'cv_f2': 0.9}, path=reg)
    df = pd.read_csv(reg)
    assert len(df) == 1
    assert df.iloc[0]['model'] == 'LR'


def test_append_registry_appends_rows(tmp_path):
    reg = tmp_path / "registry.csv"
    append_registry({'experiment': 'a', 'model': 'LR'}, path=reg)
    append_registry({'experiment': 'b', 'model': 'SVC'}, path=reg)
    df = pd.read_csv(reg)
    assert len(df) == 2
