import json
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, make_scorer
from sklearn.base import clone

# scorer for Optuna / LogisticRegressionCV / cross_val_score - needs (estimator, X, y) signature
def make_f1_scorer(n_classes: int):
    if n_classes == 2:
        return make_scorer(f1_score, average='binary', pos_label=1, zero_division=0)
    return make_scorer(f1_score, average='macro', zero_division=0)


def compute_f1(y_true, y_pred) -> float:
    # binary: F1 on toxic class (pos_label=1)
    # multiclass: F1 macro across all classes
    if len(set(y_true)) == 2:
        return round(float(f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)), 4)
    return round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4)


def _recall(y_true, y_pred) -> float:
    # binary: recall on toxic class (pos_label=1); multiclass: macro
    if len(set(y_true)) == 2:
        return round(float(recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)), 4)
    return round(float(recall_score(y_true, y_pred, average='macro', zero_division=0)), 4)


def _precision(y_true, y_pred) -> float:
    # binary: precision on toxic class (pos_label=1); multiclass: macro
    if len(set(y_true)) == 2:
        return round(float(precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)), 4)
    return round(float(precision_score(y_true, y_pred, average='macro', zero_division=0)), 4)


def compute_auc(pipe, X, y):
    # LinearSVC has decision_function but no predict_proba
    # LR and NB have predict_proba
    # returns None if AUC cannot be computed
    try:
        n = len(set(y))
        if hasattr(pipe, 'decision_function'):
            scores = pipe.decision_function(X)
            if n == 2:
                return round(float(roc_auc_score(y, scores)), 4)
            return round(float(roc_auc_score(y, scores, multi_class='ovr', average='macro')), 4)
        if hasattr(pipe, 'predict_proba'):
            proba = pipe.predict_proba(X)
            if n == 2:
                return round(float(roc_auc_score(y, proba[:, 1])), 4)
            return round(float(roc_auc_score(y, proba, multi_class='ovr', average='macro')), 4)
    except Exception:
        pass
    return None


def cv_score(pipe, X: pd.Series, y: pd.Series, cv=None) -> dict:
    # cross-validation returning F1, recall, precision
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)

    # binary: score on toxic class (pos_label=1); multiclass: macro average
    binary = len(y.unique()) == 2
    scoring = {
        'f1':        'f1'        if binary else 'f1_macro',
        'recall':    'recall'    if binary else 'recall_macro',
        'precision': 'precision' if binary else 'precision_macro',
    }

    # clone so ImbPipeline clone bug doesn't corrupt caller's pipe state
    r = cross_validate(clone(pipe), X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'cv_f1':              round(float(r['test_f1'].mean()), 4),
        'cv_f1_std':          round(float(r['test_f1'].std()), 4),
        'cv_recall':    round(float(r['test_recall'].mean()), 4),
        'cv_precision': round(float(r['test_precision'].mean()), 4),
    }


def holdout_score(pipe, X_train: pd.Series, y_train: pd.Series,
                  X_test: pd.Series, y_test: pd.Series) -> dict:
    # fit on train, evaluate on test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # per-class recall stored as JSON for error analysis in notebook 06
    classes = sorted(y_test.unique())
    per_class = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)

    return {
        'test_f1':           compute_f1(y_test, y_pred),
        'test_recall':       _recall(y_test, y_pred),
        'test_precision':    _precision(y_test, y_pred),
        'test_auc':          compute_auc(pipe, X_test, y_test),
        'per_class_recall':  json.dumps({str(c): round(float(r), 4) for c, r in zip(classes, per_class)}),
    }


def ood_score(fitted_pipe, X_ood: pd.Series, y_ood: pd.Series) -> dict:
    # evaluate pre-fitted pipe on out-of-domain data
    y_pred = fitted_pipe.predict(X_ood)
    return {
        'ood_f1':              compute_f1(y_ood, y_pred),
        'ood_recall':    _recall(y_ood, y_pred),
        'ood_precision': _precision(y_ood, y_pred),
        'ood_auc':             compute_auc(fitted_pipe, X_ood, y_ood),
    }
