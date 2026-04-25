import pytest
import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.scoring import cv_score, holdout_score, ood_score, append_registry

# Tiny synthetic text corpus for fast tests
_TEXTS = [
    "kill yourself idiot",
    "nice game well played",
    "you are trash noob",
    "good job team push",
    "report this player toxic",
    "gg wp",
] * 10
_LABELS = [1, 0, 1, 0, 1, 0] * 10


def _make_pipe():
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(min_df=1)),
        ('clf',   LogisticRegression(max_iter=200, random_state=0)),
    ])


def test_cv_score_returns_required_keys():
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    result = cv_score(_make_pipe(), pd.Series(_TEXTS), pd.Series(_LABELS),
                      cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0))
    assert 'cv_macro_f1' in result
    assert 'cv_std' in result
    assert 'cv_recall_macro' in result
    assert 'cv_precision_macro' in result


def test_test_score_returns_required_keys():
    import pandas as pd
    result = holdout_score(_make_pipe(), pd.Series(_TEXTS), pd.Series(_LABELS),
                           pd.Series(_TEXTS), pd.Series(_LABELS))
    assert 'test_macro_f1' in result
    assert 'test_weighted_f1' in result
    assert 'per_class_recall' in result
    assert 'test_precision_macro' in result
    assert 'test_recall_macro' in result
    assert 'test_auc' in result
    # per_class_recall must be a JSON string
    json.loads(result['per_class_recall'])


def test_ood_score_returns_required_keys():
    import pandas as pd
    pipe = _make_pipe()
    pipe.fit(pd.Series(_TEXTS), pd.Series(_LABELS))
    result = ood_score(pipe, pd.Series(_TEXTS), pd.Series(_LABELS))
    assert 'ood_macro_f1' in result
    assert 'ood_weighted_f1' in result
    assert 'ood_precision_macro' in result
    assert 'ood_recall_macro' in result
    assert 'ood_auc' in result


def test_append_registry_creates_file(tmp_path):
    reg = tmp_path / "registry.csv"
    row = {'experiment': 'test', 'model': 'LR', 'cv_macro_f1': 0.9}
    append_registry(row, path=reg)
    df = pd.read_csv(reg)
    assert len(df) == 1
    assert df.iloc[0]['model'] == 'LR'


def test_append_registry_appends_rows(tmp_path):
    reg = tmp_path / "registry.csv"
    append_registry({'experiment': 'a', 'model': 'LR'}, path=reg)
    append_registry({'experiment': 'b', 'model': 'SVC'}, path=reg)
    df = pd.read_csv(reg)
    assert len(df) == 2
