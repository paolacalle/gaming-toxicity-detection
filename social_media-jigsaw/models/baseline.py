"""
baseline.py — classical ML baselines for multi-label toxicity detection.

Models
------
  1. TF-IDF + Logistic Regression (MultiOutputClassifier)
  2. TF-IDF + LinearSVC            (MultiOutputClassifier)

Evaluation
----------
  - Per-label precision, recall, F1
  - Macro + micro F1
  - AUC-ROC per label + macro average

Two preprocessing variants compared side-by-side:
  - standard : no slang normalisation
  - slang    : with MLBtrio/genz-slang-dataset normalisation

Usage
-----
    python3 baseline.py
    python3 baseline.py --model logreg      # only run LogReg
    python3 baseline.py --variant standard  # only run standard preprocessing
"""

import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_df, DOMAIN_STOPWORDS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# EDA: threat is extremely rare (0.3%) — flag it in evaluation output.
# class_weight='balanced' already upweights it automatically, but recall
# will still be low due to limited positive examples.
RARE_LABELS = {'threat'}

# Combined stopword list: sklearn English + Wikipedia domain artifacts.
# EDA showed Wikipedia platform terms dominate top tokens without toxicity signal.
ALL_STOPWORDS = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))

TFIDF_WORD_PARAMS = dict(
    sublinear_tf=True,
    max_features=100_000,
    ngram_range=(1, 2),
    min_df=3,
    analyzer='word',
    # stop_words is injected at build time so ablation can swap it out
)

# Char n-grams catch obfuscated toxic words: "f*ck", "a$$", "b1tch".
# Capped at 20k features (not 50k) to keep training fast.
TFIDF_CHAR_PARAMS = dict(
    sublinear_tf=True,
    max_features=20_000,
    ngram_range=(3, 5),
    min_df=3,
    analyzer='char_wb',
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Length feature extractor
# EDA finding: toxic comments are shorter on average (123 vs 216 median words),
# so char_len and word_count carry real signal alongside TF-IDF.
# ---------------------------------------------------------------------------
class LengthFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts normalised character length and word count from text."""

    def fit(self, X, y=None):
        series = pd.Series(X)
        self.char_max_ = max(series.str.len().max(), 1)
        self.word_max_ = max(series.str.split().str.len().max(), 1)
        return self

    def transform(self, X):
        series = pd.Series(X)
        char_len   = series.str.len().fillna(0).values.astype(float) / self.char_max_
        word_count = series.str.split().str.len().fillna(0).values.astype(float) / self.word_max_
        return sp.csr_matrix(np.column_stack([char_len, word_count]))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print('Loading data...')
    train = pd.read_csv('../data/train.csv')
    test  = pd.read_csv('../data/test.csv')
    test_labels = pd.read_csv('../data/test_labels.csv')

    # Rows with label == -1 were withheld from scoring in the competition
    evaluable = test_labels[test_labels['toxic'] != -1].copy()
    eval_df = evaluable.merge(test[['id', 'comment_text']], on='id')

    print(f'  Train : {len(train):,} rows')
    print(f'  Eval  : {len(eval_df):,} / {len(test_labels):,} test rows (filtered -1 labels)')
    return train, eval_df


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------
def build_variants(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    variants: list[str],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    out = {}
    if 'standard' in variants:
        print('Preprocessing — standard...')
        out['standard'] = (
            preprocess_df(train['comment_text'], slang=False),
            preprocess_df(eval_df['comment_text'], slang=False),
        )
    if 'slang' in variants:
        print('Preprocessing — slang normalised...')
        out['slang'] = (
            preprocess_df(train['comment_text'], slang=True),
            preprocess_df(eval_df['comment_text'], slang=True),
        )
    return out


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def _feature_union(stop_words: list | None = None) -> FeatureUnion:
    """
    Three feature groups:
      - word TF-IDF  : unigrams + bigrams (100k features)
      - char TF-IDF  : 3-5 char n-grams to catch obfuscated words (20k features)
      - length feats : normalised char length + word count

    stop_words defaults to ALL_STOPWORDS (English + domain). Pass
    list(ENGLISH_STOP_WORDS) to ablate the domain stopwords contribution.
    """
    if stop_words is None:
        stop_words = ALL_STOPWORDS
    return FeatureUnion([
        ('tfidf_word', TfidfVectorizer(**TFIDF_WORD_PARAMS, stop_words=stop_words)),
        ('tfidf_char', TfidfVectorizer(**TFIDF_CHAR_PARAMS)),
        ('length',     LengthFeatureExtractor()),
    ])


def build_logreg(C: float = 0.5, stop_words: list | None = None) -> Pipeline:
    # liblinear is the recommended solver for high-dim sparse TF-IDF features.
    # C=0.5 (stronger regularisation than default 1.0) reduces overfitting on
    # the dominant non-toxic class, improving precision on minority labels.
    return Pipeline([
        ('features', _feature_union(stop_words=stop_words)),
        ('scaler',   MaxAbsScaler()),
        ('clf', MultiOutputClassifier(
            LogisticRegression(
                class_weight='balanced',
                C=C,
                max_iter=1000,
                solver='liblinear',
                random_state=RANDOM_STATE,
            )
        )),
    ])


def build_svc(C: float = 0.5, stop_words: list | None = None) -> Pipeline:
    # LinearSVC has no predict_proba; wrap with CalibratedClassifierCV for AUC.
    # method='sigmoid' (Platt scaling) is faster than the default 'isotonic'.
    # cv=2: 2 folds × 6 labels = 12 fits for calibration.
    calibrated = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', C=C, max_iter=5000, random_state=RANDOM_STATE),
        cv=2,
        method='sigmoid',
    )
    return Pipeline([
        ('features', _feature_union(stop_words=stop_words)),
        ('scaler',   MaxAbsScaler()),
        ('clf', MultiOutputClassifier(calibrated)),
    ])


MODEL_REGISTRY = {
    'logreg': ('LogisticRegression', build_logreg),
    'svc':    ('LinearSVC',          build_svc),
}


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
THRESHOLDS_GRID = np.arange(0.1, 0.9, 0.02)


def find_best_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """
    Find the per-label decision threshold that maximises F1 on a validation set.

    The default 0.5 threshold is poorly calibrated for imbalanced data:
    class_weight='balanced' pushes predicted probabilities up, so the model
    needs a higher threshold to recover precision without sacrificing too much
    recall. This is especially visible in LogReg where recall ~0.95 but
    precision is ~0.05 at threshold=0.5.

    Parameters
    ----------
    y_true  : (n_samples, n_labels) binary ground truth
    y_proba : (n_samples, n_labels) predicted probabilities

    Returns
    -------
    thresholds : (n_labels,) optimal threshold per label
    """
    thresholds = np.full(len(LABELS), 0.5)
    for i in range(len(LABELS)):
        best_f1, best_t = 0.0, 0.5
        for t in THRESHOLDS_GRID:
            preds = (y_proba[:, i] >= t).astype(int)
            f = f1_score(y_true[:, i], preds, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds[i] = best_t
    return thresholds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model: Pipeline,
    X_eval: pd.Series,
    y_eval: pd.DataFrame,
    thresholds: np.ndarray | None = None,
) -> dict:
    y_true = y_eval[LABELS].values

    try:
        proba_list = model.predict_proba(X_eval)
        y_proba = np.column_stack([p[:, 1] for p in proba_list])
    except Exception:
        y_proba = None

    # Apply per-label thresholds if provided, otherwise fall back to model default
    if thresholds is not None and y_proba is not None:
        y_pred = (y_proba >= thresholds).astype(int)
    else:
        y_pred = model.predict(X_eval)

    results: dict = {}

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(LABELS):
        # Specificity (TNR): of all truly non-toxic rows for this label,
        # how many did the model correctly leave untagged?
        tn = ((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[label] = {'precision': p[i], 'recall': r[i], 'f1': f[i],
                          'specificity': specificity}

    results['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # Clean accuracy: of rows where ALL labels are 0, what fraction did the
    # model correctly predict as entirely non-toxic (all predictions also 0)?
    clean_mask = y_true.sum(axis=1) == 0
    if clean_mask.sum() > 0:
        clean_correct = (y_pred[clean_mask].sum(axis=1) == 0).sum()
        results['clean_accuracy'] = clean_correct / clean_mask.sum()
    else:
        results['clean_accuracy'] = None

    if y_proba is not None:
        auc_scores = []
        for i, label in enumerate(LABELS):
            if y_true[:, i].sum() > 0:
                auc = roc_auc_score(y_true[:, i], y_proba[:, i])
                results[label]['auc'] = auc
                auc_scores.append(auc)
        results['macro_auc'] = float(np.mean(auc_scores)) if auc_scores else None

    return results


def print_results(
    results: dict,
    model_name: str,
    variant: str,
    thresholds: np.ndarray | None = None,
) -> None:
    print(f'\n{"="*70}')
    print(f'  {model_name}  |  preprocessing: {variant}  [tuned thresholds]')
    print(f'{"="*70}')
    thresh_header = '  Thresh' if thresholds is not None else ''
    print(f'{"Label":<24} {"Prec":>8} {"Recall":>8} {"F1":>8} {"Spec(TNR)":>10} {"AUC":>8}{thresh_header}')
    print('-' * 76)
    for i, label in enumerate(LABELS):
        m = results[label]
        auc  = f'{m["auc"]:.4f}' if 'auc' in m else '     N/A'
        spec = f'{m["specificity"]:.4f}' if 'specificity' in m else '     N/A'
        note = ' *rare*' if label in RARE_LABELS else ''
        thresh_str = f'  {thresholds[i]:.2f}' if thresholds is not None else ''
        print(f'{label + note:<24} {m["precision"]:>8.4f} {m["recall"]:>8.4f} {m["f1"]:>8.4f} {spec:>10} {auc:>8}{thresh_str}')
    print('-' * 76)
    macro_auc = f'{results["macro_auc"]:.4f}' if results.get('macro_auc') else '     N/A'
    print(f'{"MACRO F1":<24} {"":>8} {"":>8} {results["macro_f1"]:>8.4f} {"":>10} {macro_auc:>8}')
    print(f'{"MICRO F1":<24} {"":>8} {"":>8} {results["micro_f1"]:>8.4f}')
    if results.get('clean_accuracy') is not None:
        print(f'  Clean comment accuracy (all-6-label=0 rows predicted entirely clean): '
              f'{results["clean_accuracy"]:.4f}')
    if any(label in RARE_LABELS for label in LABELS):
        print('  *rare* = <0.5% positive rate; low recall expected even with balanced weights')


def print_comparison(all_results: dict, model_keys: list[str], variant_keys: list[str]) -> None:
    if len(variant_keys) < 2:
        return
    print('\n\n' + '=' * 72)
    print('SUMMARY — Micro F1 by model × preprocessing variant')
    print('=' * 72)
    header = f'{"Model":<25}' + ''.join(f'{v:>14}' for v in variant_keys)
    if 'standard' in variant_keys and 'slang' in variant_keys:
        header += f'{"delta (slang-std)":>20}'
    print(header)
    print('-' * 72)
    for key in model_keys:
        name, _ = MODEL_REGISTRY[key]
        row = f'{name:<25}'
        vals = {}
        for v in variant_keys:
            f1 = all_results.get((key, v), {}).get('micro_f1', float('nan'))
            vals[v] = f1
            row += f'{f1:>14.4f}'
        if 'standard' in vals and 'slang' in vals:
            delta = vals['slang'] - vals['standard']
            row += f'{delta:>+20.4f}'
        print(row)


def print_stopwords_comparison(ablation_results: dict, model_keys: list[str], variant_keys: list[str]) -> None:
    """Print macro F1 / macro AUC delta: with vs without domain stopwords."""
    SW_KEYS = ['with_domain_sw', 'without_domain_sw']
    print('\n\n' + '=' * 80)
    print('ABLATION — domain stopwords: with vs without')
    print('=' * 80)
    col_w = 18
    header = f'{"Model + variant":<30}' + ''.join(f'{k:>{col_w}}' for k in SW_KEYS) + f'{"delta (no-sw − sw)":>{col_w}}'
    print(header)
    print('-' * 80)
    for model_key in model_keys:
        model_name, _ = MODEL_REGISTRY[model_key]
        for variant in variant_keys:
            label = f'{model_name} [{variant}]'
            with_sw    = ablation_results.get((model_key, variant, 'with_domain_sw'),    {})
            without_sw = ablation_results.get((model_key, variant, 'without_domain_sw'), {})

            def fmt(r, key):
                v = r.get(key)
                return f'{v:.4f}' if v is not None else '     N/A'

            for metric in ('macro_f1', 'macro_auc'):
                w  = with_sw.get(metric)
                wo = without_sw.get(metric)
                delta_str = f'{wo - w:>+.4f}' if w is not None and wo is not None else '     N/A'
                print(f'  {label:<28} {metric:<10} {fmt(with_sw, metric):>{col_w}} {fmt(without_sw, metric):>{col_w}} {delta_str:>{col_w}}')
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   choices=list(MODEL_REGISTRY), default=None,
                        help='Run only this model (default: all)')
    parser.add_argument('--variant', choices=['standard', 'slang'], default=None,
                        help='Run only this preprocessing variant (default: both)')
    parser.add_argument('--ablate-stopwords', action='store_true',
                        help='Run each config twice — with and without DOMAIN_STOPWORDS — '
                             'and print a side-by-side macro F1 / AUC comparison')
    args = parser.parse_args()

    model_keys   = [args.model]   if args.model   else list(MODEL_REGISTRY)
    variant_keys = [args.variant] if args.variant else ['standard', 'slang']

    # When ablating, run two stopword configs; otherwise just the default (with domain sw).
    sw_configs: dict[str, list | None] = {'with_domain_sw': None}
    if args.ablate_stopwords:
        sw_configs['without_domain_sw'] = list(ENGLISH_STOP_WORDS)

    train, eval_df = load_data()

    # Hold out 10% of train as a validation set for threshold tuning.
    # Thresholds are found on val, then applied to the real eval set.
    # The model is still fit on the full train split.
    train_fit, train_val = train_test_split(
        train, test_size=0.1, random_state=RANDOM_STATE, stratify=train['toxic']
    )
    print(f'  Train fit: {len(train_fit):,}  |  Val (threshold tuning): {len(train_val):,}')

    variants     = build_variants(train_fit, eval_df, variant_keys)
    val_variants = build_variants(train_val, eval_df, variant_keys)

    y_train = train_fit[LABELS]
    y_val   = train_val[LABELS]
    y_eval  = eval_df[LABELS]

    all_results: dict = {}
    ablation_results: dict = {}

    for model_key in model_keys:
        model_name, model_fn = MODEL_REGISTRY[model_key]
        for variant_name, (X_train, X_eval) in variants.items():
            X_val = val_variants[variant_name][0]
            for sw_label, sw_list in sw_configs.items():
                label = f'{variant_name} / {sw_label}'
                print(f'\nTraining {model_name} [{label}]...')
                model = model_fn(stop_words=sw_list)
                model.fit(X_train, y_train)

                proba_list = model.predict_proba(X_val)
                y_val_proba = np.column_stack([p[:, 1] for p in proba_list])
                thresholds = find_best_thresholds(y_val[LABELS].values, y_val_proba)
                print(f'Optimal thresholds: { {l: round(t, 2) for l, t in zip(LABELS, thresholds)} }')

                results_tuned = evaluate(model, X_eval, y_eval, thresholds=thresholds)
                print_results(results_tuned, model_name, label, thresholds=thresholds)
                best = results_tuned

                all_results[(model_key, variant_name)] = best
                ablation_results[(model_key, variant_name, sw_label)] = best

    print_comparison(all_results, model_keys, variant_keys)
    if args.ablate_stopwords:
        print_stopwords_comparison(ablation_results, model_keys, variant_keys)


if __name__ == '__main__':
    main()
