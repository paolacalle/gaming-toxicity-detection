# Gaming Toxicity Detection — Multiclass Pipeline Design

**Date:** 2026-04-25  
**Authors:** Beibarys Nyussupov  
**Status:** Approved for implementation

---

## 1. Context & Starting Point

Binary experiment complete. Results:

| Train | Test | Macro F1 |
|-------|------|----------|
| WoT | WoT | 0.8077 |
| WoT | Dota (OOD) | 0.6953 |
| Dota | Dota | 0.8790 |
| Dota | WoT (OOD) | 0.7323 |
| WoT+Dota | WoT | 0.8132 |
| WoT+Dota | Dota | 0.8526 |

**Research question:** How many classes can we add before performance starts to drop — in-game and cross-game?

Datasets (post-cleaning, conflicting labels already dropped):
- WoT: 6 classes — 0=Non-Toxic, 1=Insults, 2=Other Offensive, 3=Hate, 4=Threats, 5=Extremism
- Dota: 4 classes — 0=Other/Non-Toxic, 1=Ego, 2=Aggression, 3=Impolite
- Processed parquets: `data/processed_data/wot/`, `data/processed_data/dota/`

---

## 2. Architecture: Option C — Shared `src/` + Thin Notebooks

### 2.1 Directory Structure

```
src/
  stopwords.py          # exists
  loaders.py            # load_wot(), load_dota(), binarise(), apply_label_scheme()
  pipelines.py          # build_pipe(clf, oversampler, tfidf_cfg) → ImbPipeline
  scoring.py            # cv_score(), test_score(), ood_score(), append_registry()
  label_schemes.py      # all label merge maps — see Section 3

notebooks/
  gaming/
    initial_cleaning_gaming.ipynb   # exists
    EDA_gaming.ipynb                # exists
  01_granularity_experiment.ipynb
  02_ml_pipeline_multiclass.ipynb
  03_anomaly_detection.ipynb
  06_error_analysis.ipynb           # runs before ensemble — see Section 7
  04_ensemble.ipynb                 # conditional on 1pp improvement
  05_comparison_dashboard.ipynb     # pure analysis, loads registry only

data/
  results/
    results_registry.csv            # append-only, one row per model eval
```

### 2.2 Results Registry Schema

```
experiment, train_game, test_game, n_classes, label_scheme,
model, cv_macro_f1, cv_std, test_macro_f1, test_weighted_f1,
per_class_recall (json string), ood_macro_f1, ood_weighted_f1,
anomaly_auroc (NULL for classification experiments),
notes, timestamp
```

`train_game` and `test_game` are separate columns. `per_class_recall` stores a JSON dict so per-class analysis is available without re-running models. `append_registry()` in `src/scoring.py` appends one row; creates file with header if missing.

**Why not just macro F1 everywhere:** False negatives (missed toxic) cost more. Per-class recall required — especially Extremism in WoT, Aggression in Dota.

---

## 3. Label Schemes (`src/label_schemes.py`)

### 3.1 Per-Dataset Incremental (in-game F1 only — no OOD beyond binary)

**WoT incremental order — by severity (easiest discrimination first):**

| n_classes | Classes | Mapping |
|-----------|---------|---------|
| 2 | Non-Toxic / Toxic | 0→0, 1+2+3+4+5→1 |
| 3 | Non-Toxic / Mild / Severe | 0→0, 2+3→1, 1+4+5→2 |
| 4 | Non-Toxic / Other Off. / Hate / Threats+Insults+Extremism | 0→0, 2→1, 3→2, 1+4+5→3 |
| 5 | Non-Toxic / Insults / Other Off. / Hate / Threats+Extremism | 0→0, 1→1, 2→2, 3→3, 4+5→4 |
| 6 | All original | identity |

Rationale for order: Extremism (5) is rarest and hardest. Threats (4) have lexical overlap with Insults (1). Grouping them last isolates the hard cases for final splits. **This ordering must be validated** in `01_granularity_experiment` with a class balance check before training.

**Dota incremental order:**

| n_classes | Mapping |
|-----------|---------|
| 2 | 0→0, 1+2+3→1 |
| 3 | 0→0, 3→1 (Impolite/mild), 1+2→2 (Ego+Aggression/severe) |
| 4 | All original |

### 3.2 Unified Cross-Game Scheme — DEFERRED

OOD at 3+ classes requires a unified label mapping across games. This mapping is non-trivial: WoT and Dota class semantics differ (e.g. WoT "Other Offensive" ≠ Dota "Impolite"). Imposing a mapping upfront risks corrupting the training signal with subjective merges.

**Decision:** Default to binary OOD (already proven, clean). After notebook 01 per-dataset results are in:
- If TF-IDF centroid clustering produces interpretable cross-game groups → design unified scheme data-driven, add notebook 01b.
- If not → document as limitation, keep OOD at binary only.

Do not implement unified 3-class scheme until centroid analysis in notebook 01 validates it.

---

## 4. Notebook 01 — Granularity Experiment

**Research question:** At what n_classes does macro F1 start to drop?

### Structure:

**Section 0:** Imports, CONFIG, load data from `src/loaders.py`

**Section 1:** Class balance check at each granularity
- Print class counts for every n_classes step for both games
- Flag any class < 500 samples (too small for 5-fold CV) — merge with nearest neighbor rather than splitting

**Section 2:** Per-dataset incremental (WoT 2→6, Dota 2→4)
- For each n_classes: run all 3 models (LR, NB, LinearSVC) + RandomOverSampler + TF-IDF. Do NOT inherit binary winner — best model may change at higher granularities (e.g. NB degrades faster with rare classes, SVC may sharpen on well-separated classes).
- 5-fold StratifiedKFold CV, record cv_macro_f1 for each model. Best model per step = row in registry.
- No OOD here — label schemes differ

**Section 4:** Centroid clustering analysis (gateway for unified scheme)
- Compute TF-IDF centroids per class per game
- Cosine similarity matrix: WoT classes × Dota classes
- If interpretable cross-game clusters emerge → define unified scheme in `label_schemes.py` and run OOD at that granularity. Append rows to registry.
- If no clear clusters → document limitation, OOD stays binary only.

**Section 5:** Sweet spot plots
- Line: in-game F1 vs n_classes (per-dataset track)
- OOD line: binary point always shown. Additional points only if unified scheme validated in Section 4.
- Annotate inflection point

All results appended to `results_registry.csv` via `append_registry()`.

---

## 5. Notebook 02 — Full ML Pipeline at Best Granularity

Runs at n_classes = sweet spot identified in 01. If sweet spot differs between in-game track and OOD track (e.g. in-game peaks at 4-class, OOD peaks at 3-class), run full pipeline at both n_classes values independently. Each run appends its own registry row with `label_scheme` indicating which track. Dashboard compares them.

### Structure:

**Section 1:** Oversampling comparison (RandomOverSampler vs SMOTE — skip BorderlineSMOTE/ADASYN, already shown inferior in binary)

**Section 2:** Model selection (LR, NB, LinearSVC) with Optuna TPE tuning

**Section 3:** Evaluate best model
- In-game CV + test set
- OOD at binary level always (binarise predictions for cross-game eval)
- OOD at best granularity only if unified scheme was validated in notebook 01 Section 4
- Per-class recall in registry

**Section 4:** Interpretability
- LR: top positive/negative TF-IDF coefficients per class (direct from `.coef_`)
- LinearSVC: same via `.coef_`
- **No SHAP for LinearSVC.** SHAP requires `predict_proba`. LinearSVC has none. Use `CalibratedClassifierCV(LinearSVC(...))` only if probability scores needed elsewhere. For interpretability, coefficient inspection is sufficient and more honest for linear models.
- Word clouds of top features per class, split by in-game correct / OOD error

**Section 5:** OOD error pattern analysis (setup for notebook 06)
- Identify top 50 OOD false negatives per class
- Print samples + their actual labels
- Note systematic patterns (document in markdown cell)

---

## 6. Notebook 03 — Anomaly Detection

**Hypothesis:** Model trained only on non-toxic text (class 0) should flag toxic text as anomalies via reconstruction/isolation score.

**Training data:** Class 0 from processed parquets (already clean — conflicting labels removed pre-split).

### Dimensionality reduction (required):
TF-IDF on text → high-dimensional sparse matrix. IsolationForest and OneClassSVM degrade in this space. Apply `TruncatedSVD(n_components=100)` (LSA) before anomaly models. 100 components: retains ~80% variance for typical text corpora, fast, sparse-compatible.

### Models:

| Model | Rationale |
|-------|-----------|
| IsolationForest | Fast, tree-based, good baseline |
| OneClassSVM (rbf) | Kernel method, better boundary for text |

Both compared. Metric: **AUROC** on toxic-vs-nontoxic separation (not F1 — no threshold needed for comparison).

### Setups (3):
1. Train WoT class 0 → score WoT val (all classes)
2. Train Dota class 0 → score Dota val (all classes)  
3. Train WoT+Dota class 0 → score each game separately

### Per-class anomaly score distribution:
Box plot of anomaly scores per original class. Expected: Non-Toxic scores low, Extremism/Aggression scores high. Deviations are interesting findings.

### Registry:
Record AUROC per setup per model. Add `anomaly_auroc` column to registry (NULL for classification experiments).

**Known limitation:** TruncatedSVD compresses domain-specific lexicon. Extremism leetspeak (`naz1`) may not survive compression into the same components as clean toxic text. Document this in notebook markdown.

---

## 7. Notebook 06 — Error Analysis (runs before ensemble)

**Why before ensemble:** If error analysis reveals a feature fix, ensemble uses improved features.

### Structure:

**Section 1:** Load best model from registry (highest `test_macro_f1` at best granularity). Also load best OOD performer (highest `ood_macro_f1`). These may differ.

**Section 2:** False negative analysis (in-game)
- Confusion matrix
- Top 100 FN samples per class, sorted by confidence
- Top TF-IDF features for FN samples vs correctly classified toxic

**Section 3:** OOD false negative analysis
- Same as Section 2 but on cross-game test set
- Compare: which classes fail most in OOD? Is it systematic vocabulary shift?

**Section 4:** Hypothesis generation
- Document patterns found (markdown cells — this is the narrative for the paper)
- Candidate fixes ranked by expected impact:
  1. Char n-grams (1,4) to catch leetspeak evasion — add to TF-IDF `analyzer='char_wb'` as secondary feature, concat with word n-grams
  2. Custom regex features for known evasion patterns from EDA (`naz1`, `d1ot`, `k1ll`)
  3. Expanding stopword list if tactical gaming terms inflate FP rate

**Section 5:** Apply fix, re-train, re-evaluate
- Implement highest-impact fix
- Re-run best model with fix
- Append improved model row to registry with `notes='char_ngram_fix'`
- If improvement < 0.5pp macro F1, document and discard fix

---

## 8. Notebook 04 — Ensemble (conditional)

**Gate:** Only proceed if stacking beats best single model by ≥ 1pp macro F1 on holdout. Otherwise document result and skip.

### Architecture: Stacking with out-of-fold meta-features

Base models: LR + LinearSVC + NB (all with best hyperparams from notebook 02)  
Meta-learner: LogisticRegression (simple, interpretable)

**Critical:** Meta-learner trains on **out-of-fold predictions only** — not in-sample. Use `cross_val_predict` with `method='predict_proba'` for LR/NB, and `CalibratedClassifierCV` for LinearSVC (needed here specifically for stacking, not interpretability).

### Evaluation:
- In-game CV macro F1
- OOD macro F1 (unified scheme)
- Compare to best single model in registry
- If Δ < 1pp: append row with `notes='ensemble_no_improvement'`, stop

---

## 9. Notebook 05 — Comparison Dashboard

Pure analysis, no training. Loads `results_registry.csv`.

### Plots:
1. Line: in-game macro F1 vs n_classes, per dataset
2. Line: OOD macro F1 vs n_classes (unified scheme only)
3. Bar: best model per experiment type
4. Heatmap: model × dataset × n_classes macro F1
5. OOD gap table: in-game minus OOD at each granularity
6. Box plot: anomaly scores per class (from notebook 03)
7. Before/after: error analysis fix improvement

---

## 10. `src/` Modules — Contracts

### `src/loaders.py`
```python
# Path anchoring — works from any caller depth (notebook or script)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_WOT  = PROJECT_ROOT / "data/processed_data/wot"
DATA_DIR_DOTA = PROJECT_ROOT / "data/processed_data/dota"

def load_wot(split: str) -> pd.DataFrame          # split = 'train' | 'val'
def load_dota(split: str) -> pd.DataFrame
def apply_label_scheme(df, scheme: dict) -> pd.DataFrame  # scheme from label_schemes.py
def load_combined(split: str, scheme: dict) -> pd.DataFrame
```

### `src/pipelines.py`
```python
def build_pipe(clf, oversampler=None, tfidf_cfg=None) -> ImbPipeline
# tfidf_cfg defaults to CONFIG['tfidf'] from binary experiment
```

### `src/scoring.py`
```python
def cv_score(pipe, X, y, cv, scoring) -> dict
def test_score(pipe, X_train, y_train, X_test, y_test) -> dict  # fits + evaluates
def ood_score(fitted_pipe, X_ood, y_ood) -> dict
def append_registry(row: dict, path: Path) -> None
```

### `src/label_schemes.py`
```python
WOT_SCHEMES: dict[int, dict]   # n_classes → {old_label: new_label}
DOTA_SCHEMES: dict[int, dict]
UNIFIED_3: dict[str, dict]     # 'wot' → mapping, 'dota' → mapping
```

---

## 11. Style Rules (per project conventions)

- Every code cell preceded by a comment explaining the why
- Every major section followed by markdown cell interpreting results
- No silent cells — every cell prints at least one output
- `CONFIG` dict at top of each notebook, no magic numbers in cells
- `src/` imports at top, never inline in cells
- Registry append at end of every model evaluation cell

---

## 12. Open Questions (to resolve in notebook 01)

1. Is any class < 500 samples at any WoT granularity step? If yes, merge that class with nearest neighbor rather than splitting.
2. Does centroid clustering in notebook 01 Section 4 reveal interpretable cross-game groups? If yes → design unified scheme data-driven and add notebook 01b. If no → binary OOD only, document as limitation.
3. Does TruncatedSVD(100) retain enough variance for anomaly detection? Print explained variance ratio in notebook 03 Section 0 — if < 70%, increase to 200.
