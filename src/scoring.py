import json
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, fbeta_score, roc_auc_score, make_scorer

# F2 scorer for Optuna: recall weighted 2x over precision.
# Missing toxic (FN) is worse than false flag (FP) in moderation.
F2_SCORER = make_scorer(fbeta_score, beta=2, average='binary', pos_label=1, zero_division=0)


def compute_f2(y_true, y_pred) -> float:
    # binary problem: F2 on toxic class (pos_label=1)
    # multiclass problem: F2 macro averaged across all classes
    if len(set(y_true)) == 2:
        return round(float(fbeta_score(y_true, y_pred, beta=2, average='binary', pos_label=1, zero_division=0)), 4)
    return round(float(fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)), 4)


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
    # cross-validation returning F2, recall macro, precision macro
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)

    # build F2 scorer that matches the class count in y
    if len(y.unique()) == 2:
        f2_cv = make_scorer(fbeta_score, beta=2, average='binary', pos_label=1, zero_division=0)
    else:
        f2_cv = make_scorer(fbeta_score, beta=2, average='macro', zero_division=0)

    scoring = {
        'f2': f2_cv,
        'recall_macro': 'recall_macro',
        'precision_macro': 'precision_macro',
    }

    r = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'cv_f2': round(float(r['test_f2'].mean()), 4),
        'cv_f2_std': round(float(r['test_f2'].std()), 4),
        'cv_recall_macro': round(float(r['test_recall_macro'].mean()), 4),
        'cv_precision_macro': round(float(r['test_precision_macro'].mean()), 4),
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
        'test_f2': compute_f2(y_test, y_pred),
        'test_recall_macro': round(float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'test_precision_macro': round(float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'test_auc': compute_auc(pipe, X_test, y_test),
        'per_class_recall': json.dumps({str(c): round(float(r), 4) for c, r in zip(classes, per_class)}),
    }


def ood_score(fitted_pipe, X_ood: pd.Series, y_ood: pd.Series) -> dict:
    # evaluate pre-fitted pipe on out-of-domain data
    y_pred = fitted_pipe.predict(X_ood)
    return {
        'ood_f2': compute_f2(y_ood, y_pred),
        'ood_recall_macro': round(float(recall_score(y_ood, y_pred, average='macro', zero_division=0)), 4),
        'ood_precision_macro': round(float(precision_score(y_ood, y_pred, average='macro', zero_division=0)), 4),
        'ood_auc': compute_auc(fitted_pipe, X_ood, y_ood),
    }


def append_registry(row: dict, path=None) -> None:
    # append one result row to the shared CSV registry
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'data' / 'results' / 'results_registry.csv'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
