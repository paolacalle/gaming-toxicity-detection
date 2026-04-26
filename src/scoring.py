import json
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              fbeta_score, roc_auc_score, make_scorer)

# ── Public scorer constant ─────────────────────────────────────────────────────
# F2 weights recall 2x over precision.
# In toxicity detection, missing a toxic message (FN) is worse than a false flag (FP).
# Binary only: pos_label=1 = toxic class. Import this into notebooks for Optuna objectives.
F2_SCORER = make_scorer(fbeta_score, beta=2, average='binary',
                         pos_label=1, zero_division=0)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, prefix: str) -> dict:
    """F2, macro-F1, macro-recall, macro-precision with a given key prefix.
    F2 uses binary average (pos_label=1) for 2-class problems, macro otherwise."""
    n = len(set(y_true))
    avg_f2 = 'binary' if n == 2 else 'macro'
    kw_f2  = {'pos_label': 1} if n == 2 else {}
    return {
        f'{prefix}f2':               round(float(fbeta_score(y_true, y_pred, beta=2,
                                                               average=avg_f2,
                                                               zero_division=0, **kw_f2)), 4),
        f'{prefix}macro_f1':         round(float(f1_score(y_true, y_pred,
                                                            average='macro', zero_division=0)), 4),
        f'{prefix}recall_macro':     round(float(recall_score(y_true, y_pred,
                                                               average='macro', zero_division=0)), 4),
        f'{prefix}precision_macro':  round(float(precision_score(y_true, y_pred,
                                                                   average='macro', zero_division=0)), 4),
    }


def _compute_auc(pipe, X, y) -> float | None:
    """ROC-AUC via decision_function (LinearSVC) or predict_proba (LR, NB).
    Returns None if neither is available or computation fails."""
    try:
        n = len(set(y))
        if hasattr(pipe, 'decision_function'):
            scores = pipe.decision_function(X)
            if n == 2:
                return round(float(roc_auc_score(y, scores)), 4)
            return round(float(roc_auc_score(y, scores,
                                              multi_class='ovr', average='macro')), 4)
        if hasattr(pipe, 'predict_proba'):
            proba = pipe.predict_proba(X)
            if n == 2:
                return round(float(roc_auc_score(y, proba[:, 1])), 4)
            return round(float(roc_auc_score(y, proba,
                                              multi_class='ovr', average='macro')), 4)
    except Exception:
        pass
    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def cv_score(pipe, X: pd.Series, y: pd.Series, cv=None) -> dict:
    """Cross-validation scoring: F2, macro-F1, macro-recall, macro-precision.
    F2 average auto-detects binary vs multiclass from y."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)

    n = len(y.unique())
    avg_f2 = 'binary' if n == 2 else 'macro'
    kw_f2  = {'pos_label': 1} if n == 2 else {}
    scoring = {
        'f2':              make_scorer(fbeta_score, beta=2, average=avg_f2,
                                       zero_division=0, **kw_f2),
        'f1_macro':        'f1_macro',
        'recall_macro':    'recall_macro',
        'precision_macro': 'precision_macro',
    }

    r = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'cv_f2':              round(float(r['test_f2'].mean()), 4),
        'cv_f2_std':          round(float(r['test_f2'].std()), 4),
        'cv_macro_f1':        round(float(r['test_f1_macro'].mean()), 4),
        'cv_recall_macro':    round(float(r['test_recall_macro'].mean()), 4),
        'cv_precision_macro': round(float(r['test_precision_macro'].mean()), 4),
    }


def holdout_score(pipe, X_train: pd.Series, y_train: pd.Series,
                   X_test: pd.Series, y_test: pd.Series) -> dict:
    """Fit on train, evaluate on test. Returns metrics + per-class recall JSON + AUC."""
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    classes = sorted(y_test.unique())
    per_class_recall = recall_score(y_test, y_pred, labels=classes,
                                     average=None, zero_division=0)
    return {
        **_compute_metrics(y_test, y_pred, prefix='test_'),
        'test_auc':         _compute_auc(pipe, X_test, y_test),
        'per_class_recall': json.dumps({str(c): round(float(r), 4)
                                        for c, r in zip(classes, per_class_recall)}),
    }


def ood_score(fitted_pipe, X_ood: pd.Series, y_ood: pd.Series) -> dict:
    """Evaluate already-fitted pipe on OOD data. Pipe must be pre-fitted."""
    y_pred = fitted_pipe.predict(X_ood)
    return {
        **_compute_metrics(y_ood, y_pred, prefix='ood_'),
        'ood_auc': _compute_auc(fitted_pipe, X_ood, y_ood),
    }


def append_registry(row: dict, path=None) -> None:
    """Append one row to results registry CSV. Creates file with header if missing."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'data' / 'results' / 'results_registry.csv'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
