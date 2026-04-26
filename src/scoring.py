import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              fbeta_score, roc_auc_score, make_scorer)

# F2 scorer: weights recall 2x over precision — missing toxic is worse than false flag.
# Auto-detects binary vs multiclass via y at call time.
def _make_f2_scorer(average: str):
    return make_scorer(fbeta_score, beta=2, average=average,
                       pos_label=1 if average == 'binary' else None,
                       zero_division=0)

def _f2(y_true, y_pred) -> float:
    average = 'binary' if len(set(y_true)) == 2 else 'macro'
    kw = {'pos_label': 1} if average == 'binary' else {}
    return round(float(fbeta_score(y_true, y_pred, beta=2, average=average,
                                    zero_division=0, **kw)), 4)

def _avg(y_true) -> str:
    return 'binary' if len(set(y_true)) == 2 else 'macro'


def cv_score(pipe, X: pd.Series, y: pd.Series, cv=None) -> dict:
    """Cross-validation with F2, F1, recall, precision. Binary or macro auto-detected."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)
    average = _avg(y)
    f2_cv = _make_f2_scorer(average)
    scoring = {
        'f2':        f2_cv,
        'f1_macro':  'f1_macro',
        'recall_macro':    'recall_macro',
        'precision_macro': 'precision_macro',
    }
    results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'cv_f2':               round(float(results['test_f2'].mean()), 4),
        'cv_f2_std':           round(float(results['test_f2'].std()), 4),
        'cv_macro_f1':         round(float(results['test_f1_macro'].mean()), 4),
        'cv_recall_macro':     round(float(results['test_recall_macro'].mean()), 4),
        'cv_precision_macro':  round(float(results['test_precision_macro'].mean()), 4),
    }


def holdout_score(pipe, X_train: pd.Series, y_train: pd.Series,
                   X_test: pd.Series, y_test: pd.Series) -> dict:
    """Fit on train, evaluate on test. Returns F2, F1, recall, precision, AUC."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # AUC
    auc = None
    try:
        n_classes = len(set(y_test))
        if hasattr(pipe, 'decision_function'):
            scores = pipe.decision_function(X_test)
            auc = round(float(roc_auc_score(y_test, scores if n_classes > 2
                                             else scores)), 4)
        elif hasattr(pipe, 'predict_proba'):
            proba = pipe.predict_proba(X_test)
            auc = round(float(roc_auc_score(
                y_test, proba[:, 1] if n_classes == 2 else proba,
                multi_class='ovr' if n_classes > 2 else 'raise',
                average='macro' if n_classes > 2 else None)), 4)
    except Exception:
        auc = None

    return {
        'test_f2':              _f2(y_test, y_pred),
        'test_macro_f1':        round(float(f1_score(y_test, y_pred, average='macro',     zero_division=0)), 4),
        'test_recall_macro':    round(float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'test_precision_macro': round(float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'test_auc':             auc,
        'per_class_recall':     json.dumps({
            str(c): round(float(r), 4)
            for c, r in zip(sorted(y_test.unique()),
                            recall_score(y_test, y_pred,
                                          labels=sorted(y_test.unique()),
                                          average=None, zero_division=0))
        }),
    }


def ood_score(fitted_pipe, X_ood: pd.Series, y_ood: pd.Series) -> dict:
    """Evaluate already-fitted pipe on OOD data."""
    y_pred = fitted_pipe.predict(X_ood)
    ood_auc = None
    try:
        n_classes = len(set(y_ood))
        if hasattr(fitted_pipe, 'decision_function'):
            scores = fitted_pipe.decision_function(X_ood)
            ood_auc = round(float(roc_auc_score(y_ood, scores)), 4)
        elif hasattr(fitted_pipe, 'predict_proba'):
            proba = fitted_pipe.predict_proba(X_ood)
            ood_auc = round(float(roc_auc_score(
                y_ood, proba[:, 1] if n_classes == 2 else proba,
                multi_class='ovr' if n_classes > 2 else 'raise',
                average='macro' if n_classes > 2 else None)), 4)
    except Exception:
        ood_auc = None

    return {
        'ood_f2':               _f2(y_ood, y_pred),
        'ood_macro_f1':         round(float(f1_score(y_ood, y_pred, average='macro',     zero_division=0)), 4),
        'ood_recall_macro':     round(float(recall_score(y_ood, y_pred, average='macro', zero_division=0)), 4),
        'ood_precision_macro':  round(float(precision_score(y_ood, y_pred, average='macro', zero_division=0)), 4),
        'ood_auc':              ood_auc,
    }


def append_registry(row: dict, path=None) -> None:
    """Append one row to results registry CSV. Creates file with header if missing."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "results" / "results_registry.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    df = pd.DataFrame([row])
    df.to_csv(path, mode='a', header=not path.exists(), index=False)
