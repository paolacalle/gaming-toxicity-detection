import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
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
    assert 'cv_f1' in result
    assert 'cv_f1_std' in result
    assert 'cv_recall' in result
    assert 'cv_precision' in result


def test_holdout_score_returns_required_keys():
    result = holdout_score(make_pipe(), pd.Series(TEXTS), pd.Series(LABELS),
                           pd.Series(TEXTS), pd.Series(LABELS))
    assert 'test_f1' in result
    assert 'test_recall' in result
    assert 'test_precision' in result
    assert 'test_auc' in result
    assert 'per_class_recall' in result
    json.loads(result['per_class_recall'])


def test_ood_score_returns_required_keys():
    pipe = make_pipe()
    pipe.fit(pd.Series(TEXTS), pd.Series(LABELS))
    result = ood_score(pipe, pd.Series(TEXTS), pd.Series(LABELS))
    assert 'ood_f1' in result
    assert 'ood_recall' in result
    assert 'ood_precision' in result
    assert 'ood_auc' in result


def make_always_one_pipe():
    # always predicts class 1: recall_class1=1.0, recall_class0=0.0
    # recall_macro=0.5 != recall_binary=1.0 — exposes the macro vs binary bug
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(min_df=1)),
        ('clf', DummyClassifier(strategy='constant', constant=1)),
    ])


def test_binary_recall_uses_class1_not_macro():
    result = holdout_score(make_always_one_pipe(), pd.Series(TEXTS), pd.Series(LABELS),
                           pd.Series(TEXTS), pd.Series(LABELS))
    # always-1 model: recall on class 1 = 1.0, recall_macro = 0.5
    assert result['test_recall'] == 1.0


def test_binary_precision_uses_class1_not_macro():
    result = holdout_score(make_always_one_pipe(), pd.Series(TEXTS), pd.Series(LABELS),
                           pd.Series(TEXTS), pd.Series(LABELS))
    # always-1 model: precision on class 1 = 0.5 (half positives), precision_macro = 0.25
    assert result['test_precision'] == 0.5


def test_ood_binary_recall_uses_class1_not_macro():
    pipe = make_always_one_pipe()
    pipe.fit(pd.Series(TEXTS), pd.Series(LABELS))
    result = ood_score(pipe, pd.Series(TEXTS), pd.Series(LABELS))
    assert result['ood_recall'] == 1.0


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
