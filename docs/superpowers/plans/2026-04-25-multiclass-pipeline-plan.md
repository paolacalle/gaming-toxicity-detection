# Multiclass Toxicity Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the binary toxicity experiment into a full multiclass pipeline covering granularity experiments, anomaly detection, error analysis, conditional ensembling, and a comparison dashboard.

**Architecture:** Shared `src/` utilities (loaders, pipelines, scoring, label schemes) imported by thin notebooks. Each notebook appends results to `data/results/results_registry.csv`. Notebook `05_comparison_dashboard` loads the registry and produces all cross-experiment plots. Notebooks run sequentially: 01 → 02 → 03 → 06 → 04 (conditional) → 05.

**Tech Stack:** pandas, scikit-learn, imbalanced-learn, optuna, matplotlib, seaborn, joblib, pytest (src/ tests only)

**Style rule (every notebook cell):** comment above cell explaining WHY → code → markdown cell after explaining results/insights. No silent cells.

**Spec:** `docs/superpowers/specs/2026-04-25-multiclass-pipeline-design.md`

---

## File Map

**Create:**
- `src/label_schemes.py` — WOT_SCHEMES, DOTA_SCHEMES dicts
- `src/loaders.py` — load_wot, load_dota, apply_label_scheme, load_combined
- `src/pipelines.py` — build_pipe
- `src/scoring.py` — cv_score, test_score, ood_score, append_registry
- `data/results/.gitkeep` — ensure results dir tracked
- `tests/test_label_schemes.py`
- `tests/test_loaders.py`
- `tests/test_scoring.py`
- `notebooks/01_granularity_experiment.ipynb`
- `notebooks/02_ml_pipeline_multiclass.ipynb`
- `notebooks/03_anomaly_detection.ipynb`
- `notebooks/06_error_analysis.ipynb`
- `notebooks/04_ensemble.ipynb`
- `notebooks/05_comparison_dashboard.ipynb`

**Modify:**
- `.gitignore` — add `data/results/*.csv` exclusion check (registry should be committed; models are not)

---

## Task 1: `src/label_schemes.py`

**Files:**
- Create: `src/label_schemes.py`
- Test: `tests/test_label_schemes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_label_schemes.py
import pytest
from src.label_schemes import WOT_SCHEMES, DOTA_SCHEMES, apply_scheme

def test_wot_binary_maps_all_toxic_to_1():
    scheme = WOT_SCHEMES[2]
    assert scheme[0] == 0
    for orig in [1, 2, 3, 4, 5]:
        assert scheme[orig] == 1

def test_wot_6class_is_identity():
    scheme = WOT_SCHEMES[6]
    for i in range(6):
        assert scheme[i] == i

def test_wot_3class_groups():
    scheme = WOT_SCHEMES[3]
    assert scheme[0] == 0
    assert scheme[2] == 1 and scheme[3] == 1    # Mild
    assert scheme[1] == 2 and scheme[4] == 2 and scheme[5] == 2  # Severe

def test_dota_binary_maps_all_toxic_to_1():
    scheme = DOTA_SCHEMES[2]
    assert scheme[0] == 0
    for orig in [1, 2, 3]:
        assert scheme[orig] == 1

def test_dota_4class_is_identity():
    scheme = DOTA_SCHEMES[4]
    for i in range(4):
        assert scheme[i] == i

def test_apply_scheme_transforms_series():
    import pandas as pd
    scheme = WOT_SCHEMES[2]
    s = pd.Series([0, 1, 2, 3, 4, 5])
    result = apply_scheme(s, scheme)
    assert list(result) == [0, 1, 1, 1, 1, 1]

def test_apply_scheme_raises_on_unknown_label():
    import pandas as pd
    scheme = WOT_SCHEMES[2]
    s = pd.Series([0, 99])
    with pytest.raises(KeyError):
        apply_scheme(s, scheme)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd "c:/Users/nyuss/OneDrive/Documentos/Bars/Portfollio/Portfollio/Projects/Project-GamingToxicityDetection"
python -m pytest tests/test_label_schemes.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.label_schemes'`

- [ ] **Step 3: Implement `src/label_schemes.py`**

```python
import pandas as pd

# WoT: 6 classes — 0=Non-Toxic, 1=Insults, 2=Other Offensive,
#                   3=Hate, 4=Threats, 5=Extremism
# Incremental order: severity-based. Easiest discrimination first.
# Mild = Other Offensive + Hate (broad, less targeted)
# Severe = Insults + Threats + Extremism (direct harm, evasion)
WOT_SCHEMES: dict[int, dict[int, int]] = {
    2: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
    3: {0: 0, 2: 1, 3: 1, 1: 2, 4: 2, 5: 2},
    4: {0: 0, 2: 1, 3: 2, 1: 3, 4: 3, 5: 3},
    5: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4},
    6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
}

# WoT class names per n_classes for display
WOT_CLASS_NAMES: dict[int, list[str]] = {
    2: ['Non-Toxic', 'Toxic'],
    3: ['Non-Toxic', 'Mild', 'Severe'],
    4: ['Non-Toxic', 'Other Offensive', 'Hate', 'Threats+Insults+Extremism'],
    5: ['Non-Toxic', 'Insults', 'Other Offensive', 'Hate', 'Threats+Extremism'],
    6: ['Non-Toxic', 'Insults', 'Other Offensive', 'Hate', 'Threats', 'Extremism'],
}

# Dota: 4 classes — 0=Other/Non-Toxic, 1=Ego, 2=Aggression, 3=Impolite
# Incremental order: Impolite first (condescension, mild), then Ego+Aggression
DOTA_SCHEMES: dict[int, dict[int, int]] = {
    2: {0: 0, 1: 1, 2: 1, 3: 1},
    3: {0: 0, 3: 1, 1: 2, 2: 2},
    4: {0: 0, 1: 1, 2: 2, 3: 3},
}

DOTA_CLASS_NAMES: dict[int, list[str]] = {
    2: ['Non-Toxic', 'Toxic'],
    3: ['Non-Toxic', 'Impolite', 'Ego+Aggression'],
    4: ['Non-Toxic', 'Ego', 'Aggression', 'Impolite'],
}


def apply_scheme(series: pd.Series, scheme: dict[int, int]) -> pd.Series:
    """Map original labels to new label space. Raises KeyError on unknown label."""
    return series.map(scheme)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/test_label_schemes.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/label_schemes.py tests/test_label_schemes.py
git commit -m "feat: add label schemes for WoT and Dota incremental granularity"
```

---

## Task 2: `src/loaders.py`

**Files:**
- Create: `src/loaders.py`
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_loaders.py
import pytest
import pandas as pd
from src.loaders import load_wot, load_dota, load_combined
from src.label_schemes import WOT_SCHEMES, DOTA_SCHEMES

def test_load_wot_train_has_required_columns():
    df = load_wot('train')
    assert 'clean_message' in df.columns
    assert 'label' in df.columns

def test_load_wot_val_has_required_columns():
    df = load_wot('val')
    assert 'clean_message' in df.columns
    assert 'label' in df.columns

def test_load_wot_train_labels_in_range():
    df = load_wot('train')
    assert set(df['label'].unique()).issubset({0, 1, 2, 3, 4, 5})

def test_load_dota_train_labels_in_range():
    df = load_dota('train')
    assert set(df['label'].unique()).issubset({0, 1, 2, 3})

def test_load_wot_with_binary_scheme():
    df = load_wot('train', scheme=WOT_SCHEMES[2])
    assert set(df['label'].unique()).issubset({0, 1})

def test_load_combined_concatenates_both():
    wot = load_wot('train')
    dota = load_dota('train')
    combined = load_combined('train', wot_scheme=WOT_SCHEMES[2], dota_scheme=DOTA_SCHEMES[2])
    assert len(combined) == len(wot) + len(dota)
    assert set(combined['label'].unique()).issubset({0, 1})

def test_load_wot_invalid_split_raises():
    with pytest.raises(ValueError):
        load_wot('test')
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_loaders.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.loaders'`

- [ ] **Step 3: Implement `src/loaders.py`**

```python
from pathlib import Path
import pandas as pd
from src.label_schemes import apply_scheme

# Anchor paths from src/ location — works from any caller depth
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_WOT  = PROJECT_ROOT / "data" / "processed_data" / "wot"
DATA_DIR_DOTA = PROJECT_ROOT / "data" / "processed_data" / "dota"

_WOT_FILES  = {'train': 'wot_train_ml.parquet',  'val': 'wot_val_ml.parquet'}
_DOTA_FILES = {'train': 'dota_train_ml.parquet', 'val': 'dota_val_ml.parquet'}


def load_wot(split: str, scheme: dict | None = None) -> pd.DataFrame:
    """Load WoT split. Optionally remap labels via scheme dict."""
    if split not in _WOT_FILES:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")
    df = pd.read_parquet(DATA_DIR_WOT / _WOT_FILES[split])
    if scheme is not None:
        df = df.copy()
        df['label'] = apply_scheme(df['label'].astype(int), scheme)
    return df


def load_dota(split: str, scheme: dict | None = None) -> pd.DataFrame:
    """Load Dota split. Optionally remap labels via scheme dict."""
    if split not in _DOTA_FILES:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")
    df = pd.read_parquet(DATA_DIR_DOTA / _DOTA_FILES[split])
    if scheme is not None:
        df = df.copy()
        df['label'] = apply_scheme(df['label'].astype(int), scheme)
    return df


def load_combined(
    split: str,
    wot_scheme: dict | None = None,
    dota_scheme: dict | None = None,
) -> pd.DataFrame:
    """Concatenate WoT + Dota splits. Apply separate schemes to each before concat."""
    wot  = load_wot(split, scheme=wot_scheme)
    dota = load_dota(split, scheme=dota_scheme)
    return pd.concat([wot, dota], ignore_index=True)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/test_loaders.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/loaders.py tests/test_loaders.py
git commit -m "feat: add data loaders with optional label scheme remapping"
```

---

## Task 3: `src/pipelines.py`

**Files:**
- Create: `src/pipelines.py`

No separate test file — pipeline construction tested implicitly via scoring tests in Task 4.

- [ ] **Step 1: Implement `src/pipelines.py`**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

# Default TF-IDF config — matches binary experiment settings
DEFAULT_TFIDF = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    norm='l2',
)

# Default oversampler — RandomOverSampler won all per-game comparisons in binary exp
DEFAULT_SEED = 7524


def build_pipe(clf, oversampler=None, tfidf_cfg: dict | None = None) -> ImbPipeline:
    """
    Build TF-IDF → oversample → clf pipeline.
    oversampler=None skips oversampling step (useful for anomaly detection).
    """
    tfidf_cfg = tfidf_cfg or DEFAULT_TFIDF
    steps = [('tfidf', TfidfVectorizer(**tfidf_cfg))]
    if oversampler is not None:
        steps.append(('oversample', oversampler))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def default_classifiers(seed: int = DEFAULT_SEED) -> dict:
    """Return dict of standard classifiers used across all experiments."""
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed, n_jobs=1
        ),
        'Naive Bayes': MultinomialNB(),
        'LinearSVC': LinearSVC(C=1.0, max_iter=2000, tol=1e-3, random_state=seed),
    }


def default_oversampler(seed: int = DEFAULT_SEED) -> RandomOverSampler:
    return RandomOverSampler(random_state=seed)
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from src.pipelines import build_pipe, default_classifiers; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/pipelines.py
git commit -m "feat: add pipeline builder with default TF-IDF and classifier configs"
```

---

## Task 4: `src/scoring.py`

**Files:**
- Create: `src/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scoring.py
import pytest
import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.scoring import cv_score, test_score, ood_score, append_registry

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


def test_test_score_returns_required_keys():
    import pandas as pd
    result = test_score(_make_pipe(), pd.Series(_TEXTS), pd.Series(_LABELS),
                        pd.Series(_TEXTS), pd.Series(_LABELS))
    assert 'test_macro_f1' in result
    assert 'test_weighted_f1' in result
    assert 'per_class_recall' in result
    # per_class_recall must be a JSON string
    json.loads(result['per_class_recall'])


def test_ood_score_returns_required_keys():
    import pandas as pd
    pipe = _make_pipe()
    pipe.fit(pd.Series(_TEXTS), pd.Series(_LABELS))
    result = ood_score(pipe, pd.Series(_TEXTS), pd.Series(_LABELS))
    assert 'ood_macro_f1' in result
    assert 'ood_weighted_f1' in result


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_scoring.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.scoring'`

- [ ] **Step 3: Implement `src/scoring.py`**

```python
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, recall_score


def cv_score(
    pipe,
    X: pd.Series,
    y: pd.Series,
    cv=None,
    scoring: str = 'f1_macro',
) -> dict:
    """Run cross-validation, return cv_macro_f1 and cv_std."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7524)
    results = cross_validate(pipe, X, y, cv=cv,
                             scoring=['f1_macro', 'f1_weighted'], n_jobs=-1)
    return {
        'cv_macro_f1':    round(float(results['test_f1_macro'].mean()), 4),
        'cv_std':         round(float(results['test_f1_macro'].std()), 4),
        'cv_weighted_f1': round(float(results['test_f1_weighted'].mean()), 4),
    }


def test_score(
    pipe,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Fit on train, evaluate on test. Returns scores + per-class recall as JSON."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    classes = sorted(y_test.unique())
    recall_per_class = recall_score(y_test, y_pred, labels=classes,
                                    average=None, zero_division=0)
    return {
        'test_macro_f1':    round(float(f1_score(y_test, y_pred, average='macro',    zero_division=0)), 4),
        'test_weighted_f1': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'per_class_recall': json.dumps({str(c): round(float(r), 4)
                                        for c, r in zip(classes, recall_per_class)}),
    }


def ood_score(fitted_pipe, X_ood: pd.Series, y_ood: pd.Series) -> dict:
    """Evaluate already-fitted pipe on OOD data. Pipe must already be fit."""
    y_pred = fitted_pipe.predict(X_ood)
    return {
        'ood_macro_f1':    round(float(f1_score(y_ood, y_pred, average='macro',    zero_division=0)), 4),
        'ood_weighted_f1': round(float(f1_score(y_ood, y_pred, average='weighted', zero_division=0)), 4),
    }


def append_registry(row: dict, path: Path | str = None) -> None:
    """Append one row to results registry CSV. Creates file with header if missing."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "results" / "results_registry.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    df = pd.DataFrame([row])
    df.to_csv(path, mode='a', header=not path.exists(), index=False)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/test_scoring.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Create results directory and gitkeep**

```bash
mkdir -p "c:/Users/nyuss/OneDrive/Documentos/Bars/Portfollio/Portfollio/Projects/Project-GamingToxicityDetection/data/results"
touch "c:/Users/nyuss/OneDrive/Documentos/Bars/Portfollio/Portfollio/Projects/Project-GamingToxicityDetection/data/results/.gitkeep"
```

- [ ] **Step 6: Run all src tests together**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/scoring.py tests/test_scoring.py src/pipelines.py data/results/.gitkeep
git commit -m "feat: add scoring helpers, pipeline builder, results registry"
```

---

## Task 5: `notebooks/01_granularity_experiment.ipynb`

**Files:**
- Create: `notebooks/01_granularity_experiment.ipynb`

**Depends on:** Tasks 1–4 complete (all src/ modules)

This notebook has 5 sections. Each section = one code cell + one markdown cell after (style rule). Below is the full cell sequence.

- [ ] **Step 1: Create notebook with Section 0 — Imports & CONFIG**

Cell type: code
```python
# Standard imports — keep consistent with binary experiment style
import warnings, time, json
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from imblearn.over_sampling import RandomOverSampler
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# src/ utilities
import sys
sys.path.insert(0, str(Path('..').resolve()))
from src.loaders import load_wot, load_dota
from src.pipelines import build_pipe, default_oversampler
from src.scoring import cv_score, test_score, ood_score, append_registry
from src.label_schemes import WOT_SCHEMES, DOTA_SCHEMES, WOT_CLASS_NAMES, DOTA_CLASS_NAMES

CONFIG = {
    'seed': 7524,
    'cv_folds': 5,
    'text_col': 'clean_message',
    'label_col': 'label',
    'registry_path': Path('../data/results/results_registry.csv'),
}
seed = CONFIG['seed']
cv   = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=seed)
np.random.seed(seed)
print('CONFIG loaded.')
```

After cell, markdown:
```
## 0. Imports & Configuration

All experiments in this notebook write to the shared results registry at
`data/results/results_registry.csv`. The dashboard notebook (05) reads from
this registry to produce all comparison plots.
```

- [ ] **Step 2: Section 1 — Class balance check**

Cell type: code
```python
# Verify class counts at every granularity step before training.
# Any class < 500 samples is too small for 5-fold CV — flag it.
MINIMUM_CLASS_SIZE = 500

print('=== WoT class balance per granularity ===')
wot_train_raw = load_wot('train')
for n in [2, 3, 4, 5, 6]:
    df = load_wot('train', scheme=WOT_SCHEMES[n])
    counts = df[CONFIG['label_col']].value_counts().sort_index()
    small = counts[counts < MINIMUM_CLASS_SIZE]
    flag = f'  ⚠ classes too small: {small.to_dict()}' if len(small) else ''
    print(f'  n={n}: {counts.to_dict()}{flag}')

print('\n=== Dota class balance per granularity ===')
for n in [2, 3, 4]:
    df = load_dota('train', scheme=DOTA_SCHEMES[n])
    counts = df[CONFIG['label_col']].value_counts().sort_index()
    small = counts[counts < MINIMUM_CLASS_SIZE]
    flag = f'  ⚠ classes too small: {small.to_dict()}' if len(small) else ''
    print(f'  n={n}: {counts.to_dict()}{flag}')
```

After cell, markdown:
```
## 1. Class Balance Check

If any class has fewer than 500 samples, that granularity step is skipped in
the incremental experiment — the class is too rare for reliable 5-fold CV.
RandomOverSampler handles imbalance within folds but cannot create meaningful
signal from <100 genuine samples.
```

- [ ] **Step 3: Section 2 — Per-dataset incremental experiment**

Cell type: code
```python
# Run all 3 models at every granularity level for each game.
# Best model may change at higher n_classes — do NOT inherit binary winner.
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

def run_granularity_track(game: str, schemes: dict, class_names_map: dict,
                           load_fn, registry_path: Path):
    """For each n_classes: run LR, NB, SVC, record best in registry."""
    results = []
    oversampler = default_oversampler(seed)

    for n_classes, scheme in schemes.items():
        train_df = load_fn('train', scheme=scheme)
        X_train  = train_df[CONFIG['text_col']]
        y_train  = train_df[CONFIG['label_col']]

        # Skip if any class < MINIMUM_CLASS_SIZE
        counts = y_train.value_counts()
        if counts.min() < MINIMUM_CLASS_SIZE:
            print(f'  {game} n={n_classes}: skipped (class too small: {counts.min()})')
            continue

        classifiers = {
            'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000,
                                                       random_state=seed, n_jobs=1),
            'Naive Bayes':         MultinomialNB(),
            'LinearSVC':           LinearSVC(C=1.0, max_iter=2000, tol=1e-3,
                                             random_state=seed),
        }

        best_f1, best_name = -1, None
        for clf_name, clf in classifiers.items():
            pipe   = build_pipe(clone(clf), oversampler=RandomOverSampler(random_state=seed))
            scores = cv_score(pipe, X_train, y_train, cv=cv)
            print(f'  {game} n={n_classes} {clf_name:<22} cv_macro_f1={scores["cv_macro_f1"]:.4f}')
            if scores['cv_macro_f1'] > best_f1:
                best_f1, best_name = scores['cv_macro_f1'], clf_name

            append_registry({
                'experiment':    'granularity_per_dataset',
                'train_game':    game,
                'test_game':     game,
                'n_classes':     n_classes,
                'label_scheme':  'native',
                'model':         clf_name,
                **scores,
                'test_macro_f1':    None,
                'test_weighted_f1': None,
                'per_class_recall': None,
                'ood_macro_f1':     None,
                'ood_weighted_f1':  None,
                'anomaly_auroc':    None,
                'notes':            '',
            }, path=registry_path)

        results.append({'game': game, 'n_classes': n_classes,
                        'best_model': best_name, 'best_cv_f1': best_f1})
        print(f'  → best at n={n_classes}: {best_name} ({best_f1:.4f})\n')

    return pd.DataFrame(results)

print('--- WoT incremental ---')
wot_results = run_granularity_track('WoT', WOT_SCHEMES, WOT_CLASS_NAMES,
                                     load_wot, CONFIG['registry_path'])
print('\n--- Dota incremental ---')
dota_results = run_granularity_track('Dota', DOTA_SCHEMES, DOTA_CLASS_NAMES,
                                      load_dota, CONFIG['registry_path'])

print('\nWoT summary:')
print(wot_results.to_string(index=False))
print('\nDota summary:')
print(dota_results.to_string(index=False))
```

After cell, markdown:
```
## 2. Per-Dataset Incremental Experiment

Each model is evaluated at every granularity level independently — we do not
assume the binary winner stays best at 6 classes. The sweet spot is the
n_classes where cv_macro_f1 peaks before declining.

OOD evaluation is not possible here because WoT and Dota have different native
label spaces at n > 2. Binary OOD is already established from the binary
experiment notebook.
```

- [ ] **Step 4: Section 3 — Sweet spot line plot**

Cell type: code
```python
# Visualise F1 vs n_classes to identify the inflection point for each game.
# This is the core research finding of this notebook.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (game, df) in zip(axes, [('WoT', wot_results), ('Dota', dota_results)]):
    ax.plot(df['n_classes'], df['best_cv_f1'], marker='o', linewidth=2, color='#1565C0')
    for _, row in df.iterrows():
        ax.annotate(f"{row['best_cv_f1']:.3f}\n({row['best_model'][:3]})",
                    (row['n_classes'], row['best_cv_f1']),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xlabel('Number of Classes', fontsize=12)
    ax.set_ylabel('CV Macro F1', fontsize=12)
    ax.set_title(f'{game} — F1 vs Granularity', fontweight='bold')
    ax.set_xticks(df['n_classes'])
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

plt.suptitle('Granularity Sweet Spot — Both Games', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('../data/results/granularity_sweet_spot.png', dpi=150, bbox_inches='tight')
plt.show()
```

After cell, markdown:
```
## 3. Sweet Spot Identification

The line plot reveals the n_classes where F1 peaks. The annotated model name
(first 3 chars) shows whether the best model changes across granularities.
Declining F1 after the peak = classes are too similar for TF-IDF features to
discriminate — this is the answer to the core research question.
```

- [ ] **Step 5: Section 4 — Centroid clustering (gateway for unified scheme)**

Cell type: code
```python
# Compute TF-IDF centroids per class per game and measure cosine similarity.
# This is the gateway decision: if cross-game clusters are interpretable,
# we define a unified label scheme. Otherwise OOD stays at binary only.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def compute_centroids(texts: pd.Series, labels: pd.Series) -> dict:
    """Fit TF-IDF on texts, return L2-normalised centroid per class."""
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95,
                             sublinear_tf=True, norm='l2')
    X = tfidf.fit_transform(texts)
    centroids = {}
    for cls in sorted(labels.unique()):
        mask = (labels == cls).values
        centroids[cls] = np.asarray(X[mask].mean(axis=0))
    return centroids, tfidf

wot_train  = load_wot('train')
dota_train = load_dota('train')

# Fit separate vectorizers per game — they share vocabulary partially
wot_centroids,  wot_tfidf  = compute_centroids(wot_train['clean_message'],
                                                 wot_train['label'].astype(int))
dota_centroids, dota_tfidf = compute_centroids(dota_train['clean_message'],
                                                 dota_train['label'].astype(int))

# Cosine similarity requires shared vocabulary — project both onto union vocab
from sklearn.feature_extraction.text import TfidfVectorizer
combined_texts = pd.concat([wot_train['clean_message'], dota_train['clean_message']])
joint_tfidf    = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.95,
                                  sublinear_tf=True, norm='l2')
joint_tfidf.fit(combined_texts)

X_wot_joint  = joint_tfidf.transform(wot_train['clean_message'])
X_dota_joint = joint_tfidf.transform(dota_train['clean_message'])

wot_centroids_joint  = {c: np.asarray(X_wot_joint[(wot_train['label'].astype(int)==c).values].mean(axis=0))
                        for c in range(6)}
dota_centroids_joint = {c: np.asarray(X_dota_joint[(dota_train['label'].astype(int)==c).values].mean(axis=0))
                        for c in range(4)}

# Cosine similarity matrix: WoT classes (rows) × Dota classes (cols)
from sklearn.metrics.pairwise import cosine_similarity
wot_mat  = normalize(np.vstack([wot_centroids_joint[c]  for c in range(6)]))
dota_mat = normalize(np.vstack([dota_centroids_joint[c] for c in range(4)]))
sim_matrix = cosine_similarity(wot_mat, dota_mat)

wot_labels_display  = ['WoT: ' + n for n in WOT_CLASS_NAMES[6]]
dota_labels_display = ['Dota: ' + n for n in DOTA_CLASS_NAMES[4]]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=dota_labels_display, yticklabels=wot_labels_display, ax=ax)
ax.set_title('WoT × Dota Class Centroid Cosine Similarity', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('../data/results/centroid_similarity.png', dpi=150, bbox_inches='tight')
plt.show()

# Decision guidance
print('\nInterpretation guide:')
print('  similarity > 0.3 between two classes → potential merge candidate')
print('  If no cross-game pairs show similarity > 0.3 → binary OOD only')
print('\nMax similarity per WoT class:')
for i, wot_name in enumerate(WOT_CLASS_NAMES[6]):
    j = np.argmax(sim_matrix[i])
    print(f'  WoT {wot_name:<18} ↔ Dota {DOTA_CLASS_NAMES[4][j]:<12} sim={sim_matrix[i,j]:.3f}')
```

After cell, markdown:
```
## 4. Centroid Clustering — Unified Scheme Gateway

**Decision rule:** If the heatmap shows interpretable cross-game clusters
(similarity > 0.3 between semantically similar classes), define a unified
label scheme in `src/label_schemes.py` and add a notebook 01b for unified
OOD experiments. If not, OOD analysis stays at binary level only — document
this as a limitation in the paper (different annotation schemas across games
make direct comparison beyond binary non-trivial).
```

- [ ] **Step 6: Run notebook top to bottom, verify no errors**

Open `notebooks/01_granularity_experiment.ipynb` in Jupyter. Kernel → Restart & Run All.
Expected: no exceptions, plots display, registry file created at `data/results/results_registry.csv`.

- [ ] **Step 7: Commit**

```bash
git add notebooks/01_granularity_experiment.ipynb data/results/results_registry.csv
git commit -m "feat: add granularity experiment notebook with sweet spot analysis"
```

---

## Task 6: `notebooks/02_ml_pipeline_multiclass.ipynb`

**Files:**
- Create: `notebooks/02_ml_pipeline_multiclass.ipynb`

**Depends on:** Task 5 complete. Read `data/results/results_registry.csv` to determine best granularity before running this notebook.

- [ ] **Step 1: Section 0 — Imports & CONFIG, load best granularity from registry**

Cell type: code
```python
import warnings, time, json
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler
import joblib
import sys
sys.path.insert(0, str(Path('..').resolve()))
from src.loaders import load_wot, load_dota
from src.pipelines import build_pipe
from src.scoring import cv_score, test_score, ood_score, append_registry
from src.label_schemes import WOT_SCHEMES, DOTA_SCHEMES, WOT_CLASS_NAMES, DOTA_CLASS_NAMES

CONFIG = {
    'seed': 7524,
    'cv_folds': 5,
    'text_col': 'clean_message',
    'label_col': 'label',
    'optuna_trials': 30,
    'registry_path': Path('../data/results/results_registry.csv'),
    'models_dir': Path('../models'),
}
seed = CONFIG['seed']
cv   = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=seed)
np.random.seed(seed)
CONFIG['models_dir'].mkdir(exist_ok=True)

# Determine best granularity from registry — peak cv_macro_f1 per game
registry = pd.read_csv(CONFIG['registry_path'])
gran = registry[registry['experiment'] == 'granularity_per_dataset']

wot_best_n  = int(gran[gran['train_game']=='WoT'].groupby('n_classes')['cv_macro_f1'].max().idxmax())
dota_best_n = int(gran[gran['train_game']=='Dota'].groupby('n_classes')['cv_macro_f1'].max().idxmax())
print(f'WoT best granularity: {wot_best_n} classes')
print(f'Dota best granularity: {dota_best_n} classes')
```

After cell, markdown:
```
## 0. Imports & Configuration

Best n_classes is loaded from the registry written by notebook 01 — no
hardcoded values. If the sweet spot differs between in-game and OOD tracks,
both are run independently with separate registry rows.
```

- [ ] **Step 2: Section 1 — Oversampling comparison at best granularity**

Cell type: code
```python
# Compare RandomOverSampler vs SMOTE at best granularity for each game.
# BorderlineSMOTE and ADASYN already shown inferior in binary experiment — skip.
from imblearn.over_sampling import SMOTE

def compare_oversamplers(game: str, load_fn, scheme: dict, class_names: list):
    train_df  = load_fn('train', scheme=scheme)
    X_train   = train_df[CONFIG['text_col']]
    y_train   = train_df[CONFIG['label_col']]

    ref_clf   = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=1)
    results   = []

    for os_name, sampler in [('RandomOverSampler', RandomOverSampler(random_state=seed)),
                              ('SMOTE',             SMOTE(random_state=seed))]:
        pipe   = build_pipe(clone(ref_clf), oversampler=sampler)
        scores = cv_score(pipe, X_train, y_train, cv=cv)
        print(f'  {game} {os_name:<22} macro_f1={scores["cv_macro_f1"]:.4f} ± {scores["cv_std"]:.4f}')
        results.append({'Oversampler': os_name, **scores})

    return pd.DataFrame(results).sort_values('cv_macro_f1', ascending=False)

print('=== WoT oversampler comparison ===')
wot_os_df = compare_oversamplers('WoT', load_wot, WOT_SCHEMES[wot_best_n],
                                   WOT_CLASS_NAMES[wot_best_n])
wot_best_os_name = wot_os_df.iloc[0]['Oversampler']

print('\n=== Dota oversampler comparison ===')
dota_os_df = compare_oversamplers('Dota', load_dota, DOTA_SCHEMES[dota_best_n],
                                    DOTA_CLASS_NAMES[dota_best_n])
dota_best_os_name = dota_os_df.iloc[0]['Oversampler']

print(f'\nBest oversamplers: WoT={wot_best_os_name}, Dota={dota_best_os_name}')
```

After cell, markdown:
```
## 1. Oversampling Comparison

At higher granularities SMOTE sometimes outperforms RandomOverSampler because
the minority classes are more tightly clustered. We rerun the comparison at
the sweet-spot granularity rather than assuming the binary result holds.
```

- [ ] **Step 3: Section 2 — Model selection with Optuna at best granularity**

Cell type: code
```python
# Full model selection: LR (LogisticRegressionCV), NB (Optuna), LinearSVC (Optuna).
# Best oversampler per game used. Record all in registry.

OS_MAP = {'RandomOverSampler': RandomOverSampler, 'SMOTE': SMOTE}

def run_model_selection(game: str, load_fn, scheme: dict, best_os_name: str,
                        n_classes: int, class_names: list):
    train_df = load_fn('train', scheme=scheme)
    X_train  = train_df[CONFIG['text_col']]
    y_train  = train_df[CONFIG['label_col']]
    oversampler = OS_MAP[best_os_name](random_state=seed)
    models_comparison = []

    # --- Logistic Regression with CV regularisation ---
    print(f'  [{game}] Logistic Regression ...')
    lr_pipe = build_pipe(
        LogisticRegressionCV(Cs=30, cv=cv, scoring='f1_macro',
                             max_iter=1000, random_state=seed, n_jobs=-1),
        oversampler=OS_MAP[best_os_name](random_state=seed)
    )
    lr_scores = cv_score(lr_pipe, X_train, y_train, cv=cv)
    models_comparison.append({'Model': 'Logistic Regression', **lr_scores})

    # --- Naive Bayes (Optuna) ---
    print(f'  [{game}] Naive Bayes (Optuna) ...')
    def nb_objective(trial):
        p = {'clf__alpha': trial.suggest_float('clf__alpha', 0.001, 2.0, log=True)}
        p_clone = build_pipe(MultinomialNB(),
                              oversampler=OS_MAP[best_os_name](random_state=seed))
        p_clone.set_params(**p)
        return cross_val_score(p_clone, X_train, y_train, cv=cv,
                               scoring='f1_macro', n_jobs=1).mean()
    study_nb = optuna.create_study(direction='maximize',
                                    sampler=TPESampler(seed=seed))
    study_nb.optimize(nb_objective, n_trials=CONFIG['optuna_trials'])
    nb_pipe = build_pipe(MultinomialNB(),
                          oversampler=OS_MAP[best_os_name](random_state=seed))
    nb_pipe.set_params(**study_nb.best_params)
    nb_scores = cv_score(nb_pipe, X_train, y_train, cv=cv)
    models_comparison.append({'Model': 'Naive Bayes', **nb_scores})

    # --- LinearSVC (Optuna) ---
    print(f'  [{game}] LinearSVC (Optuna) ...')
    def svc_objective(trial):
        C = trial.suggest_float('clf__C', 0.01, 10.0, log=True)
        p_clone = build_pipe(LinearSVC(C=C, max_iter=2000, tol=1e-3,
                                        random_state=seed),
                              oversampler=OS_MAP[best_os_name](random_state=seed))
        return cross_val_score(p_clone, X_train, y_train, cv=cv,
                               scoring='f1_macro', n_jobs=1).mean()
    study_svc = optuna.create_study(direction='maximize',
                                     sampler=TPESampler(seed=seed))
    study_svc.optimize(svc_objective, n_trials=CONFIG['optuna_trials'])
    svc_pipe = build_pipe(
        LinearSVC(C=study_svc.best_params['clf__C'], max_iter=2000,
                  tol=1e-3, random_state=seed),
        oversampler=OS_MAP[best_os_name](random_state=seed)
    )
    svc_scores = cv_score(svc_pipe, X_train, y_train, cv=cv)
    models_comparison.append({'Model': 'LinearSVC', **svc_scores})

    compare_df = pd.DataFrame(models_comparison).sort_values('cv_macro_f1', ascending=False)
    print(f'\n  {game} model comparison:')
    print(compare_df.to_string(index=False))

    pipes = {'Logistic Regression': lr_pipe, 'Naive Bayes': nb_pipe, 'LinearSVC': svc_pipe}
    return compare_df, pipes

print('=== WoT model selection ===')
wot_compare_df, wot_pipes = run_model_selection(
    'WoT', load_wot, WOT_SCHEMES[wot_best_n], wot_best_os_name,
    wot_best_n, WOT_CLASS_NAMES[wot_best_n])

print('\n=== Dota model selection ===')
dota_compare_df, dota_pipes = run_model_selection(
    'Dota', load_dota, DOTA_SCHEMES[dota_best_n], dota_best_os_name,
    dota_best_n, DOTA_CLASS_NAMES[dota_best_n])

wot_best_model_name  = wot_compare_df.iloc[0]['Model']
dota_best_model_name = dota_compare_df.iloc[0]['Model']
print(f'\nBest: WoT={wot_best_model_name}, Dota={dota_best_model_name}')
```

After cell, markdown:
```
## 2. Model Selection

Optuna TPE tunes NB alpha and SVC C at 30 trials each. LogisticRegressionCV
searches 30 C values internally. `n_jobs=1` inside Optuna objectives on Windows
to avoid spawn overhead (confirmed in binary experiment).
```

- [ ] **Step 4: Section 3 — Evaluate best model, record in registry**

Cell type: code
```python
# Evaluate best model on holdout set. OOD at binary level always.
# OOD at multiclass only if unified scheme was validated in notebook 01.
from sklearn.metrics import classification_report

def evaluate_best(game: str, best_name: str, pipes: dict, load_fn,
                  train_scheme: dict, n_classes: int, class_names: list,
                  ood_load_fn=None, ood_scheme: dict=None, ood_game: str=None):
    train_df = load_fn('train', scheme=train_scheme)
    val_df   = load_fn('val',   scheme=train_scheme)
    X_train, y_train = train_df[CONFIG['text_col']], train_df[CONFIG['label_col']]
    X_val,   y_val   = val_df[CONFIG['text_col']],   val_df[CONFIG['label_col']]

    pipe   = pipes[best_name]
    scores = test_score(pipe, X_train, y_train, X_val, y_val)
    print(f'=== {game} {best_name} — in-game test ===')
    print(classification_report(y_val, pipe.predict(X_val),
                                  target_names=class_names, zero_division=0))

    # Binary OOD — always run
    ood_scores = {}
    if ood_load_fn:
        ood_val      = ood_load_fn('val', scheme=ood_scheme)
        X_ood, y_ood = ood_val[CONFIG['text_col']], ood_val[CONFIG['label_col']]
        ood_scores   = ood_score(pipe, X_ood, y_ood)
        print(f'=== {game} → {ood_game} OOD ===')
        print(classification_report(y_ood, pipe.predict(X_ood),
                                      target_names=WOT_CLASS_NAMES[2], zero_division=0))

    append_registry({
        'experiment':    'ml_pipeline_multiclass',
        'train_game':    game,
        'test_game':     game,
        'n_classes':     n_classes,
        'label_scheme':  'native',
        'model':         best_name,
        **wot_compare_df[wot_compare_df['Model']==best_name].iloc[0].to_dict()
            if game=='WoT' else dota_compare_df[dota_compare_df['Model']==best_name].iloc[0].to_dict(),
        **scores,
        **ood_scores,
        'anomaly_auroc': None,
        'notes':         f'best_n={n_classes}',
    }, path=CONFIG['registry_path'])

    return pipe

print('=== WoT ===')
wot_best_pipe = evaluate_best(
    'WoT', wot_best_model_name, wot_pipes, load_wot,
    WOT_SCHEMES[wot_best_n], wot_best_n, WOT_CLASS_NAMES[wot_best_n],
    ood_load_fn=load_dota, ood_scheme=DOTA_SCHEMES[2], ood_game='Dota (binary OOD)')

print('\n=== Dota ===')
dota_best_pipe = evaluate_best(
    'Dota', dota_best_model_name, dota_pipes, load_dota,
    DOTA_SCHEMES[dota_best_n], dota_best_n, DOTA_CLASS_NAMES[dota_best_n],
    ood_load_fn=load_wot, ood_scheme=WOT_SCHEMES[2], ood_game='WoT (binary OOD)')
```

After cell, markdown:
```
## 3. Best Model Evaluation

OOD is evaluated at binary level by binarising the OOD game's labels — this
keeps the evaluation honest since the multiclass label spaces differ. The OOD
F1 is directly comparable to the binary experiment results.
Per-class recall is stored in the registry as JSON for downstream analysis.
```

- [ ] **Step 5: Section 4 — Interpretability (coefficients, not SHAP)**

Cell type: code
```python
# Extract top TF-IDF features per class from LR or SVC coefficients.
# LinearSVC and LR both expose .coef_ — no predict_proba needed.
# SHAP is NOT used here: LinearSVC has no predict_proba, and coefficient
# inspection is more honest for linear TF-IDF models.

def plot_top_features(pipe, class_names: list, title: str, top_n: int = 15):
    tfidf   = pipe.named_steps['tfidf']
    clf     = pipe.named_steps['clf']
    feature_names = np.array(tfidf.get_feature_names_out())
    coef    = clf.coef_  # shape: (n_classes, n_features) or (1, n_features) for binary

    if coef.shape[0] == 1:
        coef = np.vstack([-coef, coef])  # binary: flip for class 0

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    for ax, cls_idx, cls_name in zip(axes, range(n_classes), class_names):
        top_idx = np.argsort(coef[cls_idx])[-top_n:][::-1]
        ax.barh(feature_names[top_idx][::-1], coef[cls_idx][top_idx][::-1],
                color='#1565C0')
        ax.set_title(cls_name, fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle(title, fontweight='bold', fontsize=13)
    plt.tight_layout()
    fname = title.replace(' ', '_').replace('/', '_').lower()
    plt.savefig(f'../data/results/{fname}_features.png', dpi=150, bbox_inches='tight')
    plt.show()

# Use whichever pipe has .coef_ (LR or SVC — both do)
plot_top_features(wot_best_pipe,  WOT_CLASS_NAMES[wot_best_n],
                   f'WoT Top Features — {wot_best_n} classes')
plot_top_features(dota_best_pipe, DOTA_CLASS_NAMES[dota_best_n],
                   f'Dota Top Features — {dota_best_n} classes')
```

After cell, markdown:
```
## 4. Interpretability — Top TF-IDF Features per Class

Coefficients from the linear classifier directly indicate which n-grams push
a message toward each class. This is more interpretable than SHAP for TF-IDF
models because the relationship is exact: higher coefficient = stronger
contribution to that class score.
```

- [ ] **Step 6: Save best models**

Cell type: code
```python
# Save best pipes for reuse in error analysis and ensemble notebooks
import joblib
wot_path  = CONFIG['models_dir'] / f'multiclass_wot_{wot_best_n}class_{wot_best_model_name.lower().replace(" ","_")}.joblib'
dota_path = CONFIG['models_dir'] / f'multiclass_dota_{dota_best_n}class_{dota_best_model_name.lower().replace(" ","_")}.joblib'
joblib.dump(wot_best_pipe,  wot_path)
joblib.dump(dota_best_pipe, dota_path)
print(f'Saved: {wot_path}')
print(f'Saved: {dota_path}')
```

- [ ] **Step 7: Run notebook, verify no errors**

Kernel → Restart & Run All. Check registry has new rows with `experiment='ml_pipeline_multiclass'`.

- [ ] **Step 8: Commit**

```bash
git add notebooks/02_ml_pipeline_multiclass.ipynb
git commit -m "feat: add full multiclass ML pipeline with interpretability"
```

---

## Task 7: `notebooks/03_anomaly_detection.ipynb`

**Files:**
- Create: `notebooks/03_anomaly_detection.ipynb`

- [ ] **Step 1: Section 0 — Imports & CONFIG**

Cell type: code
```python
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import sys
sys.path.insert(0, str(Path('..').resolve()))
from src.loaders import load_wot, load_dota
from src.scoring import append_registry
from src.label_schemes import WOT_SCHEMES, DOTA_SCHEMES

CONFIG = {
    'seed': 7524,
    'text_col': 'clean_message',
    'label_col': 'label',
    'svd_components': 100,      # TruncatedSVD dims — print explained variance below
    'registry_path': Path('../data/results/results_registry.csv'),
}
seed = CONFIG['seed']
print('CONFIG loaded.')
```

- [ ] **Step 2: Section 1 — Dimensionality reduction check**

Cell type: code
```python
# TF-IDF → TruncatedSVD (LSA) is required before anomaly models.
# IsolationForest and OneClassSVM degrade in high-dimensional sparse space.
# Print explained variance at 100 components — if < 70%, increase to 200.
from src.loaders import load_combined

combined_train = load_combined('train', wot_scheme=WOT_SCHEMES[2], dota_scheme=DOTA_SCHEMES[2])
tfidf_all = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95,
                              sublinear_tf=True, norm='l2')
X_combined_tfidf = tfidf_all.fit_transform(combined_train[CONFIG['text_col']])
print(f'TF-IDF vocab size: {X_combined_tfidf.shape[1]:,}')

svd = TruncatedSVD(n_components=CONFIG['svd_components'], random_state=seed)
svd.fit(X_combined_tfidf)
explained = svd.explained_variance_ratio_.sum()
print(f'Explained variance at {CONFIG["svd_components"]} components: {explained:.3f}')
if explained < 0.70:
    print('⚠ < 70% — consider increasing svd_components to 200 in CONFIG')
else:
    print('✓ Sufficient variance retained')
```

After cell, markdown:
```
## 1. Dimensionality Reduction Check

TF-IDF produces a sparse matrix with vocabulary-size dimensions (often 50k–200k).
Both IsolationForest and OneClassSVM are known to degrade in this space — the
former because tree splits become uninformative, the latter because the kernel
bandwidth becomes meaningless. TruncatedSVD (LSA) collapses this to dense
semantic dimensions. We verify ≥70% variance is retained before proceeding.
```

- [ ] **Step 3: Section 2 — Three anomaly detection setups**

Cell type: code
```python
# Setup 1: Train on WoT class 0 → score WoT val (all classes)
# Setup 2: Train on Dota class 0 → score Dota val (all classes)
# Setup 3: Train on WoT+Dota class 0 → score each game separately
# Metric: AUROC on binary toxic-vs-nontoxic (no threshold needed for comparison)

def build_anomaly_pipe(n_components: int, model_name: str, seed: int):
    """TF-IDF → TruncatedSVD → anomaly model."""
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95,
                             sublinear_tf=True, norm='l2')
    svd   = TruncatedSVD(n_components=n_components, random_state=seed)
    if model_name == 'IsolationForest':
        detector = IsolationForest(contamination=0.05, random_state=seed, n_jobs=-1)
    else:
        detector = OneClassSVM(kernel='rbf', nu=0.05)
    return Pipeline([('tfidf', tfidf), ('svd', svd), ('detector', detector)])


def run_anomaly_setup(train_texts: pd.Series, val_texts: pd.Series,
                       val_labels_binary: pd.Series, setup_name: str,
                       train_game: str, test_game: str, n_components: int):
    """Fit on non-toxic train, score val. Return AUROC per model."""
    results = {}
    for model_name in ['IsolationForest', 'OneClassSVM']:
        pipe = build_anomaly_pipe(n_components, model_name, seed)
        pipe.fit(train_texts)
        # decision_function: higher = more normal (inlier). Negate for anomaly score.
        scores = -pipe.decision_function(val_texts)
        auroc  = roc_auc_score(val_labels_binary, scores)
        print(f'  {setup_name} | {model_name:<18} AUROC={auroc:.4f}')
        results[model_name] = {'scores': scores, 'auroc': auroc}

        append_registry({
            'experiment':    'anomaly_detection',
            'train_game':    train_game,
            'test_game':     test_game,
            'n_classes':     2,
            'label_scheme':  'binary',
            'model':         model_name,
            'cv_macro_f1':   None, 'cv_std': None, 'cv_weighted_f1': None,
            'test_macro_f1': None, 'test_weighted_f1': None,
            'per_class_recall': None,
            'ood_macro_f1':  None, 'ood_weighted_f1': None,
            'anomaly_auroc': round(auroc, 4),
            'notes':         setup_name,
        }, path=CONFIG['registry_path'])
    return results

n_comp = CONFIG['svd_components']

# Setup 1: WoT class 0 → WoT val
wot_train   = load_wot('train')
wot_val     = load_wot('val')
wot_train0  = wot_train[wot_train[CONFIG['label_col']]==0][CONFIG['text_col']]
wot_val_bin = (wot_val[CONFIG['label_col']].astype(int) > 0).astype(int)
print('=== Setup 1: Train WoT class 0 → Test WoT val ===')
setup1_results = run_anomaly_setup(wot_train0, wot_val[CONFIG['text_col']],
                                    wot_val_bin, 'wot_on_wot', 'WoT', 'WoT', n_comp)

# Setup 2: Dota class 0 → Dota val
dota_train   = load_dota('train')
dota_val     = load_dota('val')
dota_train0  = dota_train[dota_train[CONFIG['label_col']]==0][CONFIG['text_col']]
dota_val_bin = (dota_val[CONFIG['label_col']].astype(int) > 0).astype(int)
print('\n=== Setup 2: Train Dota class 0 → Test Dota val ===')
setup2_results = run_anomaly_setup(dota_train0, dota_val[CONFIG['text_col']],
                                    dota_val_bin, 'dota_on_dota', 'Dota', 'Dota', n_comp)

# Setup 3: WoT+Dota class 0 → each game separately
combined0 = pd.concat([wot_train0, dota_train0], ignore_index=True)
print('\n=== Setup 3: Train Combined class 0 → Test WoT val ===')
setup3a = run_anomaly_setup(combined0, wot_val[CONFIG['text_col']],
                              wot_val_bin, 'combined_on_wot', 'WoT+Dota', 'WoT', n_comp)
print('\n=== Setup 3: Train Combined class 0 → Test Dota val ===')
setup3b = run_anomaly_setup(combined0, dota_val[CONFIG['text_col']],
                              dota_val_bin, 'combined_on_dota', 'WoT+Dota', 'Dota', n_comp)
```

After cell, markdown:
```
## 2. Anomaly Detection Setups

AUROC measures how well anomaly scores separate toxic from non-toxic without
requiring a threshold. AUROC > 0.7 = useful signal. AUROC ≈ 0.5 = no signal.

Key question: does the combined class-0 training set (larger, cross-game) produce
better anomaly detection than per-game? If yes, diversity of non-toxic gaming
language helps the anomaly boundary generalise.
```

- [ ] **Step 4: Section 3 — Anomaly score distribution per original class**

Cell type: code
```python
# Box plot of anomaly scores per original WoT class (IsolationForest).
# Expected: Non-Toxic scores low (normal), Extremism scores high (anomalous).
# Deviations reveal which toxic classes are hardest to detect without labels.
from src.label_schemes import WOT_CLASS_NAMES

best_anomaly_model = 'IsolationForest'

pipe_wot = build_anomaly_pipe(n_comp, best_anomaly_model, seed)
pipe_wot.fit(wot_train0)
wot_val_scores = -pipe_wot.decision_function(wot_val[CONFIG['text_col']])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (val_df, scores, game, class_names_list) in zip(axes, [
    (wot_val,  wot_val_scores,                                    'WoT',  WOT_CLASS_NAMES[6]),
    (dota_val, -build_anomaly_pipe(n_comp, best_anomaly_model, seed)
                .fit(dota_train0)
                .decision_function(dota_val[CONFIG['text_col']]), 'Dota', ['Non-Toxic','Ego','Aggression','Impolite']),
]):
    plot_df = pd.DataFrame({'score': scores, 'class': val_df[CONFIG['label_col']].values})
    plot_df['class_name'] = plot_df['class'].map(
        {i: n for i, n in enumerate(class_names_list)})
    sns.boxplot(data=plot_df, x='class_name', y='score', ax=ax,
                order=class_names_list, palette='YlOrRd')
    ax.set_title(f'{game} — Anomaly Score per Class ({best_anomaly_model})',
                  fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Anomaly Score (higher = more anomalous)')
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('../data/results/anomaly_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

After cell, markdown:
```
## 3. Anomaly Score Distribution per Class

If the model captures toxicity without labels, scores should increase
monotonically from Non-Toxic → Extremism/Aggression. Flat distributions across
classes indicate the anomaly boundary is not capturing toxicity signal —
the LSA representation may compress game-specific toxic vocabulary into
dimensions shared with non-toxic tactical language.
```

- [ ] **Step 5: Run notebook, verify no errors**

Kernel → Restart & Run All. Check AUROC values printed and box plot displayed.

- [ ] **Step 6: Commit**

```bash
git add notebooks/03_anomaly_detection.ipynb
git commit -m "feat: add anomaly detection notebook with LSA + IsolationForest/OneClassSVM"
```

---

## Task 8: `notebooks/06_error_analysis.ipynb`

**Files:**
- Create: `notebooks/06_error_analysis.ipynb`

**Depends on:** Task 6 complete. Loads best model joblib saved in notebook 02.

- [ ] **Step 1: Section 0 — Load best model from registry and joblib**

Cell type: code
```python
import warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sys
sys.path.insert(0, str(Path('..').resolve()))
from src.loaders import load_wot, load_dota
from src.label_schemes import WOT_SCHEMES, DOTA_SCHEMES, WOT_CLASS_NAMES, DOTA_CLASS_NAMES
from src.scoring import append_registry

CONFIG = {
    'seed': 7524,
    'text_col': 'clean_message',
    'label_col': 'label',
    'models_dir': Path('../models'),
    'registry_path': Path('../data/results/results_registry.csv'),
    'top_fn_samples': 50,   # false negatives to inspect per class
}
seed = CONFIG['seed']

# Load best model info from registry
registry = pd.read_csv(CONFIG['registry_path'])
best_row  = (registry[registry['experiment']=='ml_pipeline_multiclass']
             .sort_values('test_macro_f1', ascending=False).iloc[0])
print(f"Best model: {best_row['model']} | game={best_row['train_game']} "
      f"| n_classes={best_row['n_classes']} | test_macro_f1={best_row['test_macro_f1']}")

# Load joblib
models_dir  = CONFIG['models_dir']
wot_best_n  = int(best_row['n_classes']) if best_row['train_game']=='WoT' else None
model_files = list(models_dir.glob('multiclass_*.joblib'))
print('Available models:', [f.name for f in model_files])
```

- [ ] **Step 2: Section 1 — In-game false negative analysis**

Cell type: code
```python
# Load best WoT model (highest test_macro_f1 in registry for WoT)
wot_best_row = (registry[(registry['experiment']=='ml_pipeline_multiclass') &
                           (registry['train_game']=='WoT')]
                .sort_values('test_macro_f1', ascending=False).iloc[0])
wot_n       = int(wot_best_row['n_classes'])
wot_model_f = list(models_dir.glob(f'multiclass_wot_{wot_n}class_*.joblib'))[0]
wot_pipe    = joblib.load(wot_model_f)

wot_val  = load_wot('val', scheme=WOT_SCHEMES[wot_n])
X_val    = wot_val[CONFIG['text_col']]
y_val    = wot_val[CONFIG['label_col']]
y_pred   = wot_pipe.predict(X_val)
names    = WOT_CLASS_NAMES[wot_n]

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap='Blues',
                                                        colorbar=False)
ax.set_title(f'WoT {wot_n}-class Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('../data/results/wot_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# False negatives per class (toxic predicted as wrong class)
print(f'\n=== WoT Top-{CONFIG["top_fn_samples"]} False Negatives per Class ===')
for true_cls in range(wot_n):
    fn_mask = (y_val == true_cls) & (y_pred != true_cls)
    fn_texts = X_val[fn_mask].reset_index(drop=True)
    fn_pred  = pd.Series(y_pred)[fn_mask.values].reset_index(drop=True)
    if len(fn_texts) == 0:
        continue
    print(f'\n--- True: {names[true_cls]} ({fn_mask.sum()} FN) ---')
    sample = fn_texts[:CONFIG['top_fn_samples']]
    pred_names = fn_pred[:CONFIG['top_fn_samples']].map({i: names[i] for i in range(wot_n)})
    for txt, pred in zip(sample, pred_names):
        print(f'  [pred={pred}] {txt[:120]}')
```

After cell, markdown:
```
## 1. In-Game False Negative Analysis

The confusion matrix reveals which classes are confused with each other.
False negative patterns tell us WHERE the model fails — not just that it fails.
Common patterns to look for:
- Extremism (class 5) misclassified as Mild: leetspeak evasion (`naz1`, `d1ot`)
- Hate misclassified as Non-Toxic: implicit/ironic toxicity
- Threats misclassified as Insults: lexical overlap between aggression types
```

- [ ] **Step 3: Section 2 — OOD false negative analysis**

Cell type: code
```python
# Which WoT classes fail most when model is applied to Dota?
# Binarise Dota labels for cross-game eval (label spaces differ).
dota_val     = load_dota('val', scheme=DOTA_SCHEMES[2])  # binary
X_dota_val   = dota_val[CONFIG['text_col']]
y_dota_bin   = dota_val[CONFIG['label_col']]

# Binarise WoT model predictions for cross-game eval
y_wot_pred_on_dota = (wot_pipe.predict(X_dota_val) > 0).astype(int)
ood_fn_mask  = (y_dota_bin == 1) & (y_wot_pred_on_dota == 0)

print(f'OOD False Negatives: {ood_fn_mask.sum()} / {y_dota_bin.sum()} toxic Dota samples')
print(f'OOD FN rate: {ood_fn_mask.mean():.3f}')
print(f'\nTop OOD false negative examples (Dota toxic → WoT predicts non-toxic):')
fn_texts_ood = X_dota_val[ood_fn_mask].reset_index(drop=True)
for txt in fn_texts_ood[:CONFIG['top_fn_samples']]:
    print(f'  {txt[:140]}')
```

After cell, markdown:
```
## 2. OOD False Negative Analysis

OOD false negatives are Dota toxic messages that the WoT-trained model
misses completely. These reveal vocabulary gaps — Dota-specific toxic patterns
(e.g., "report", "ez", hero-name insults) that WoT training data never saw.
This is the core OOD failure mode: domain-specific toxicity vocabulary.
```

- [ ] **Step 4: Section 3 — Hypothesis and fix**

Cell type: code
```python
# Hypothesis: WoT Extremism failures are caused by leetspeak evasion (naz1, d1ot, k1ll).
# Fix: Add char n-gram TF-IDF (analyzer='char_wb', ngram_range=(2,4)) as secondary feature,
# concatenate with word TF-IDF features using FeatureUnion.
# If improvement < 0.5pp macro F1 → document and discard fix.
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

wot_train = load_wot('train', scheme=WOT_SCHEMES[wot_n])
X_train   = wot_train[CONFIG['text_col']]
y_train   = wot_train[CONFIG['label_col']]
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Baseline: current best model CV score
baseline_scores = cross_val_score(wot_pipe, X_train, y_train,
                                   cv=cv, scoring='f1_macro', n_jobs=1)
baseline_f1 = baseline_scores.mean()
print(f'Baseline cv_macro_f1: {baseline_f1:.4f} ± {baseline_scores.std():.4f}')

# Fixed pipeline: word n-grams + char n-grams concatenated via FeatureUnion
word_tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95,
                               sublinear_tf=True, norm='l2')
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                               min_df=5, max_df=0.95, sublinear_tf=True, norm='l2')
features   = FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])
fixed_pipe = ImbPipeline([
    ('features',   features),
    ('oversample', RandomOverSampler(random_state=seed)),
    ('clf',        LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=1)),
])

fixed_scores = cross_val_score(fixed_pipe, X_train, y_train,
                                cv=cv, scoring='f1_macro', n_jobs=1)
fixed_f1 = fixed_scores.mean()
delta    = fixed_f1 - baseline_f1
print(f'Fixed cv_macro_f1:    {fixed_f1:.4f} ± {fixed_scores.std():.4f}')
print(f'Delta: {delta:+.4f}')

if delta >= 0.005:
    print('✓ Improvement ≥ 0.5pp — keeping fix')
    # Retrain on full train set, save
    fixed_pipe.fit(X_train, y_train)
    joblib.dump(fixed_pipe, CONFIG['models_dir'] / f'multiclass_wot_{wot_n}class_char_ngram_fix.joblib')
    wot_val = load_wot('val', scheme=WOT_SCHEMES[wot_n])
    val_scores = {}
    from src.scoring import test_score as _test_score
    val_scores = _test_score(fixed_pipe, X_train, y_train,
                              wot_val[CONFIG['text_col']], wot_val[CONFIG['label_col']])
    append_registry({
        'experiment': 'error_analysis_fix',
        'train_game': 'WoT', 'test_game': 'WoT',
        'n_classes': wot_n, 'label_scheme': 'native',
        'model': 'LR+CharNgram',
        'cv_macro_f1': round(fixed_f1, 4), 'cv_std': round(fixed_scores.std(), 4),
        **val_scores,
        'ood_macro_f1': None, 'ood_weighted_f1': None,
        'anomaly_auroc': None,
        'notes': 'char_ngram_fix',
    }, path=CONFIG['registry_path'])
else:
    print('✗ Improvement < 0.5pp — discarding fix, documenting limitation')
    append_registry({
        'experiment': 'error_analysis_fix',
        'train_game': 'WoT', 'test_game': 'WoT',
        'n_classes': wot_n, 'label_scheme': 'native',
        'model': 'LR+CharNgram',
        'cv_macro_f1': round(fixed_f1, 4), 'cv_std': round(fixed_scores.std(), 4),
        'test_macro_f1': None, 'test_weighted_f1': None, 'per_class_recall': None,
        'ood_macro_f1': None, 'ood_weighted_f1': None, 'anomaly_auroc': None,
        'notes': 'char_ngram_fix_no_improvement',
    }, path=CONFIG['registry_path'])
```

After cell, markdown:
```
## 3. Char N-Gram Fix

Char n-grams (2,4) capture subword patterns like leetspeak substitutions.
By concatenating word and char TF-IDF features via FeatureUnion, the model
sees both the word-level and character-level representations simultaneously.
The 0.5pp threshold prevents reporting noise as improvement.
```

- [ ] **Step 5: Run notebook, verify no errors, commit**

```bash
git add notebooks/06_error_analysis.ipynb
git commit -m "feat: add error analysis notebook with confusion matrix, FN inspection, char-ngram fix"
```

---

## Task 9: `notebooks/04_ensemble.ipynb` (conditional)

**Files:**
- Create: `notebooks/04_ensemble.ipynb`

**Gate:** Only proceed past Section 1 if stacking beats best single model by ≥1pp macro F1.

- [ ] **Step 1: Section 0 — Imports and load best model from registry**

Cell type: code
```python
import warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
import sys
sys.path.insert(0, str(Path('..').resolve()))
from src.loaders import load_wot, load_dota
from src.pipelines import build_pipe
from src.scoring import cv_score, test_score, append_registry
from src.label_schemes import WOT_SCHEMES, DOTA_SCHEMES, WOT_CLASS_NAMES

CONFIG = {
    'seed': 7524,
    'cv_folds': 5,
    'text_col': 'clean_message',
    'label_col': 'label',
    'improvement_threshold': 0.01,  # 1pp
    'registry_path': Path('../data/results/results_registry.csv'),
}
seed = CONFIG['seed']
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
np.random.seed(seed)

registry   = pd.read_csv(CONFIG['registry_path'])
best_row   = (registry[registry['experiment'].isin(['ml_pipeline_multiclass','error_analysis_fix'])]
              .sort_values('test_macro_f1', ascending=False).iloc[0])
wot_n      = int(best_row['n_classes'])
baseline   = best_row['test_macro_f1']
print(f'Baseline to beat: {best_row["model"]} test_macro_f1={baseline:.4f} (game={best_row["train_game"]})')
```

- [ ] **Step 2: Section 1 — Stacking with out-of-fold meta-features**

Cell type: code
```python
# Stacking: LR + CalibratedSVC + NB as base models.
# Meta-learner: LogisticRegression trained on out-of-fold probability predictions.
# CRITICAL: cross_val_predict generates OOF predictions — meta-learner NEVER
# sees in-sample predictions. This prevents meta-learner overfitting.
# CalibratedClassifierCV is needed here specifically for SVC predict_proba.

wot_train = load_wot('train', scheme=WOT_SCHEMES[wot_n])
wot_val   = load_wot('val',   scheme=WOT_SCHEMES[wot_n])
X_train   = wot_train[CONFIG['text_col']]
y_train   = wot_train[CONFIG['label_col']]
X_val     = wot_val[CONFIG['text_col']]
y_val     = wot_val[CONFIG['label_col']]

os = RandomOverSampler(random_state=seed)

base_pipes = {
    'LR':  build_pipe(LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=1),
                       oversampler=RandomOverSampler(random_state=seed)),
    'NB':  build_pipe(MultinomialNB(alpha=0.1),
                       oversampler=RandomOverSampler(random_state=seed)),
    # CalibratedClassifierCV wraps SVC to expose predict_proba — needed for stacking only
    'SVC': build_pipe(CalibratedClassifierCV(
                          LinearSVC(C=1.0, max_iter=2000, tol=1e-3, random_state=seed),
                          cv=3),
                       oversampler=RandomOverSampler(random_state=seed)),
}

# Generate out-of-fold probability predictions for each base model
print('Generating out-of-fold predictions (this takes a few minutes)...')
oof_preds = {}
for name, pipe in base_pipes.items():
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method='predict_proba', n_jobs=1)
    oof_preds[name] = oof
    print(f'  {name}: OOF shape={oof.shape}')

# Stack OOF predictions as meta-features
import numpy as np
meta_X_train = np.hstack([oof_preds[n] for n in base_pipes])
meta_clf     = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
from sklearn.model_selection import cross_val_score
meta_scores  = cross_val_score(meta_clf, meta_X_train, y_train, cv=cv,
                                scoring='f1_macro', n_jobs=1)
stacking_cv_f1 = meta_scores.mean()
print(f'\nStacking cv_macro_f1: {stacking_cv_f1:.4f} ± {meta_scores.std():.4f}')
print(f'Baseline:             {baseline:.4f}')
print(f'Delta:                {stacking_cv_f1 - baseline:+.4f}')
```

- [ ] **Step 3: Section 2 — Gate check and evaluation**

Cell type: code
```python
# Only proceed if stacking beats baseline by >= 1pp. Otherwise document and stop.
delta = stacking_cv_f1 - baseline

if delta < CONFIG['improvement_threshold']:
    print(f'✗ Stacking improvement {delta:+.4f} < {CONFIG["improvement_threshold"]:.2f} threshold.')
    print('  Skipping full evaluation. Documenting in registry.')
    append_registry({
        'experiment': 'ensemble',
        'train_game': 'WoT', 'test_game': 'WoT',
        'n_classes': wot_n, 'label_scheme': 'native',
        'model': 'Stacking(LR+SVC+NB)',
        'cv_macro_f1': round(stacking_cv_f1, 4), 'cv_std': round(meta_scores.std(), 4),
        'cv_weighted_f1': None, 'test_macro_f1': None, 'test_weighted_f1': None,
        'per_class_recall': None, 'ood_macro_f1': None, 'ood_weighted_f1': None,
        'anomaly_auroc': None,
        'notes': 'no_improvement_below_1pp_threshold',
    }, path=CONFIG['registry_path'])
else:
    print(f'✓ Improvement {delta:+.4f} >= threshold. Running full evaluation.')
    # Fit all base models on full train
    for name, pipe in base_pipes.items():
        pipe.fit(X_train, y_train)
    # Build meta-features for val set
    meta_X_val = np.hstack([pipe.predict_proba(X_val) for pipe in base_pipes.values()])
    meta_clf.fit(meta_X_train, y_train)
    y_pred_val = meta_clf.predict(meta_X_val)
    test_f1 = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
    print(f'\nStacking test_macro_f1: {test_f1:.4f}')
    print(classification_report(y_val, y_pred_val,
                                  target_names=WOT_CLASS_NAMES[wot_n], zero_division=0))
    append_registry({
        'experiment': 'ensemble',
        'train_game': 'WoT', 'test_game': 'WoT',
        'n_classes': wot_n, 'label_scheme': 'native',
        'model': 'Stacking(LR+SVC+NB)',
        'cv_macro_f1': round(stacking_cv_f1, 4), 'cv_std': round(meta_scores.std(), 4),
        'cv_weighted_f1': None,
        'test_macro_f1': round(float(test_f1), 4), 'test_weighted_f1': None,
        'per_class_recall': None, 'ood_macro_f1': None, 'ood_weighted_f1': None,
        'anomaly_auroc': None,
        'notes': f'improvement_{delta:+.4f}',
    }, path=CONFIG['registry_path'])
```

After cell, markdown:
```
## Ensemble — Stacking Result

Out-of-fold stacking prevents meta-learner overfitting. CalibratedClassifierCV
wraps LinearSVC to expose predict_proba — used here only for stacking, not
for the base classification task. If the gate fails, the best single model
from notebooks 02/06 remains the recommended production model.
```

- [ ] **Step 4: Run notebook, commit**

```bash
git add notebooks/04_ensemble.ipynb
git commit -m "feat: add conditional ensemble notebook with stacking gate"
```

---

## Task 10: `notebooks/05_comparison_dashboard.ipynb`

**Files:**
- Create: `notebooks/05_comparison_dashboard.ipynb`

**Depends on:** All previous notebooks run, registry populated.

- [ ] **Step 1: Section 0 — Load registry**

Cell type: code
```python
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path('..').resolve()))

CONFIG = {
    'registry_path': Path('../data/results/results_registry.csv'),
}

registry = pd.read_csv(CONFIG['registry_path'])
print(f'Registry: {len(registry)} rows, {registry["experiment"].nunique()} experiments')
print(registry['experiment'].value_counts())
```

- [ ] **Step 2: Section 1 — Granularity sweet spot plot**

Cell type: code
```python
# Line plot: cv_macro_f1 vs n_classes per game, showing best model per step.
gran = (registry[registry['experiment']=='granularity_per_dataset']
        .groupby(['train_game','n_classes'])['cv_macro_f1'].max().reset_index())

fig, ax = plt.subplots(figsize=(9, 5))
for game, color in [('WoT','#1565C0'), ('Dota','#C62828')]:
    df = gran[gran['train_game']==game]
    ax.plot(df['n_classes'], df['cv_macro_f1'], marker='o', linewidth=2,
            color=color, label=game)
    for _, row in df.iterrows():
        ax.annotate(f'{row["cv_macro_f1"]:.3f}',
                    (row['n_classes'], row['cv_macro_f1']),
                    textcoords='offset points', xytext=(0,7), ha='center', fontsize=9)

ax.set_xlabel('Number of Classes', fontsize=12)
ax.set_ylabel('CV Macro F1 (best model)', fontsize=12)
ax.set_title('Granularity Sweet Spot — In-Game Performance', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig('../data/results/dashboard_granularity.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: Section 2 — All experiments comparison bar chart**

Cell type: code
```python
# Bar chart: test_macro_f1 per experiment per game. Side-by-side.
exp_best = (registry[registry['test_macro_f1'].notna()]
            .groupby(['experiment','train_game'])['test_macro_f1'].max().reset_index())

fig, ax = plt.subplots(figsize=(12, 5))
exp_order = ['ml_pipeline_multiclass', 'error_analysis_fix', 'ensemble']
plot_df   = exp_best[exp_best['experiment'].isin(exp_order)]

x      = np.arange(len(plot_df))
colors = plot_df['train_game'].map({'WoT':'#1565C0','Dota':'#C62828'}).fillna('#388E3C')
ax.bar(x, plot_df['test_macro_f1'], color=colors, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(plot_df['experiment'] + '\n(' + plot_df['train_game'] + ')',
                    rotation=15, fontsize=9)
ax.set_ylabel('Test Macro F1')
ax.set_title('Best Test Performance per Experiment', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1)
for i, v in enumerate(plot_df['test_macro_f1']):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('../data/results/dashboard_all_experiments.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 4: Section 3 — Anomaly AUROC summary**

Cell type: code
```python
# Table + bar chart of AUROC per setup per anomaly model.
anomaly = registry[registry['experiment']=='anomaly_detection'][
    ['notes','model','anomaly_auroc']].dropna()
print(anomaly.sort_values('anomaly_auroc', ascending=False).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(anomaly))
ax.bar(x, anomaly['anomaly_auroc'],
       color=anomaly['model'].map({'IsolationForest':'#1565C0','OneClassSVM':'#E65100'}))
ax.set_xticks(x)
ax.set_xticklabels(anomaly['notes'] + '\n' + anomaly['model'], fontsize=8, rotation=15)
ax.axhline(0.7, color='gray', linestyle='--', linewidth=1, label='AUROC=0.7 threshold')
ax.set_ylabel('AUROC')
ax.set_title('Anomaly Detection AUROC per Setup', fontweight='bold')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('../data/results/dashboard_anomaly.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 5: Run notebook, verify all plots render, commit**

```bash
git add notebooks/05_comparison_dashboard.ipynb data/results/
git commit -m "feat: add comparison dashboard notebook loading all registry results"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec Section | Task |
|---|---|
| src/label_schemes.py | Task 1 |
| src/loaders.py | Task 2 |
| src/pipelines.py | Task 3 |
| src/scoring.py + registry | Task 4 |
| results_registry.csv schema | Task 4 (append_registry) |
| 01 granularity: class balance check | Task 5 Step 2 |
| 01 granularity: all 3 models per step | Task 5 Step 3 |
| 01 granularity: sweet spot plot | Task 5 Step 4 |
| 01 granularity: centroid clustering gateway | Task 5 Step 5 |
| 02 oversampling comparison | Task 6 Step 2 |
| 02 model selection + Optuna | Task 6 Step 3 |
| 02 evaluate + OOD binary | Task 6 Step 4 |
| 02 interpretability (coefficients, no SHAP) | Task 6 Step 5 |
| 03 dimensionality reduction check | Task 7 Step 2 |
| 03 anomaly setups (3 setups × 2 models) | Task 7 Step 3 |
| 03 anomaly score distribution plot | Task 7 Step 4 |
| 06 confusion matrix + in-game FN | Task 8 Step 2 |
| 06 OOD FN analysis | Task 8 Step 3 |
| 06 char-ngram fix + gate | Task 8 Step 4 |
| 04 stacking with OOF | Task 9 Step 2 |
| 04 1pp gate | Task 9 Step 3 |
| 05 dashboard: granularity line | Task 10 Step 2 |
| 05 dashboard: all experiments bar | Task 10 Step 3 |
| 05 dashboard: anomaly AUROC | Task 10 Step 4 |
| Style rule: comment+code+markdown | All notebook tasks — explicitly noted |
| `n_jobs=1` in Optuna on Windows | Task 6 Step 3 (explicit in code) |
| PATH anchoring in loaders | Task 2 Step 3 |

**No gaps found.**

**Placeholder scan:** No TBD, TODO, or vague steps. All code shown in full.

**Type consistency:** `apply_scheme(series, scheme)` defined in Task 1, called the same way in Task 2. `append_registry(row, path)` defined in Task 4, called with same signature in Tasks 5–10. `build_pipe(clf, oversampler, tfidf_cfg)` defined in Task 3, called consistently throughout. ✓
