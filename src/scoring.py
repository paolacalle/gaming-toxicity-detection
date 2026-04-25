import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score, roc_auc_score


# cross-validation score 
def cv_score(pipe, X: pd.Series, y: pd.Series, cv=None, scoring: str = 'f1_macro') -> dict:
    """Run cross-validation, return cv_macro_f1 and cv_std."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)
    results = cross_validate(pipe, X, y, cv=cv,
                             scoring=['f1_macro', 'f1_weighted', 'recall_macro', 'precision_macro'], n_jobs=-1)
    return {
        'cv_macro_f1':       round(float(results['test_f1_macro'].mean()), 4),
        'cv_std':            round(float(results['test_f1_macro'].std()), 4),
        'cv_weighted_f1':    round(float(results['test_f1_weighted'].mean()), 4),
        'cv_recall_macro':   round(float(results['test_recall_macro'].mean()), 4),
        'cv_precision_macro': round(float(results['test_precision_macro'].mean()), 4),
    }

# test score 
def holdout_score(pipe, X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series) -> dict:
    """Fit on train, evaluate on test. Returns scores + per-class recall as JSON."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    classes = sorted(y_test.unique())
    recall_per_class = recall_score(y_test, y_pred, labels=classes,
                                    average=None, zero_division=0)
    # AUC
    auc = None
    try:
        n_classes = len(classes)
        if hasattr(pipe, 'decision_function'):
            scores = pipe.decision_function(X_test)
            if n_classes == 2:
                auc = round(float(roc_auc_score(y_test, scores)), 4)
            else:
                auc = round(float(roc_auc_score(y_test, scores, multi_class='ovr', average='macro')), 4)
        elif hasattr(pipe, 'predict_proba'):
            proba = pipe.predict_proba(X_test)
            if n_classes == 2:
                auc = round(float(roc_auc_score(y_test, proba[:, 1])), 4)
            else:
                auc = round(float(roc_auc_score(y_test, proba, multi_class='ovr', average='macro')), 4)
    except Exception:
        auc = None
    return {
        'test_macro_f1':       round(float(f1_score(y_test, y_pred, average='macro',    zero_division=0)), 4),
        'test_weighted_f1':    round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'per_class_recall':    json.dumps({str(c): round(float(r), 4)
                                           for c, r in zip(classes, recall_per_class)}),
        'test_precision_macro': round(float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'test_recall_macro':   round(float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'test_auc':            auc,
    }

# out of sample testing 
def ood_score(fitted_pipe, X_ood: pd.Series, y_ood: pd.Series) -> dict:
    """Evaluate already-fitted pipe on OOD data. Pipe must already be fit."""
    y_pred = fitted_pipe.predict(X_ood)
    classes = sorted(y_ood.unique())
    # AUC
    ood_auc = None
    try:
        n_classes = len(classes)
        if hasattr(fitted_pipe, 'decision_function'):
            scores = fitted_pipe.decision_function(X_ood)
            if n_classes == 2:
                ood_auc = round(float(roc_auc_score(y_ood, scores)), 4)
            else:
                ood_auc = round(float(roc_auc_score(y_ood, scores, multi_class='ovr', average='macro')), 4)
        elif hasattr(fitted_pipe, 'predict_proba'):
            proba = fitted_pipe.predict_proba(X_ood)
            if n_classes == 2:
                ood_auc = round(float(roc_auc_score(y_ood, proba[:, 1])), 4)
            else:
                ood_auc = round(float(roc_auc_score(y_ood, proba, multi_class='ovr', average='macro')), 4)
    except Exception:
        ood_auc = None
    return {
        'ood_macro_f1':       round(float(f1_score(y_ood, y_pred, average='macro',    zero_division=0)), 4),
        'ood_weighted_f1':    round(float(f1_score(y_ood, y_pred, average='weighted', zero_division=0)), 4),
        'ood_precision_macro': round(float(precision_score(y_ood, y_pred, average='macro', zero_division=0)), 4),
        'ood_recall_macro':   round(float(recall_score(y_ood, y_pred, average='macro', zero_division=0)), 4),
        'ood_auc':            ood_auc,
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
