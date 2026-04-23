# Gaming Toxicity Detection

Detecting toxic behavior in gaming chat using NLP — with a focus on **out-of-domain (OOD) generalization** from general social media to gaming-specific language.

## Project Overview

This project investigates how well toxicity classifiers trained on general social media data transfer to gaming contexts, and why they fail when they do (slang-heavy text, implicit toxicity, absence of explicit slurs, sarcasm).

## Datasets

> Datasets to be documented.

## Project Structure

```
data/
  raw/          # Original datasets (not committed)
  processed/    # Cleaned and merged splits
notebooks/      # EDA and experiment notebooks
src/            # Source code
```

## Team

Beibarys Nyussupov, Ruby, Paola, Jett

## Roadmap

- **Week 1** — EDA, data pipeline, class imbalance handling, LR / Naïve Bayes / SVM baselines
- **Week 2** — BERT fine-tuning, cross-domain evaluation, error analysis
- **Week 3** — Results write-up, OOD narrative, paper submission

## Setup

```bash
pip install -r requirements.txt
```

## Generic ML Pipeline

The notebook pipeline has been turned into a reusable script at [src/generic_pipeline.py](/Users/paolacalle/Desktop/NYU/semesters/spring-2026/ML/hw/gaming-toxicity-detection/src/generic_pipeline.py).

Example:

```bash
python src/generic_pipeline.py \
  --train-path data/processed_data/wot/wot_train_ml.parquet \
  --test-path data/processed_data/wot/wot_val_ml.parquet \
  --output-model models/wot_generic.joblib \
  --output-report reports/wot_generic.json \
  --text-col clean_message \
  --label-col label \
  --use-custom-stopwords
```

Use `--binary-threshold 0` to collapse a multiclass label into binary toxicity (`label > 0`).

## Generic Anomaly Pipeline

For novelty-style detection, use [src/generic_anomaly_pipeline.py](/Users/paolacalle/Desktop/NYU/semesters/spring-2026/ML/hw/gaming-toxicity-detection/src/generic_anomaly_pipeline.py).

This version does:
- normal-only training
- normal-only threshold tuning
- mixed test evaluation where any label other than the normal label is treated as an anomaly

Example:

```bash
python src/generic_anomaly_pipeline.py \
  --train-path data/processed_data/wot/wot_train_ml.parquet \
  --tune-path data/processed_data/wot/wot_val_ml.parquet \
  --test-path data/processed_data/dota/dota_val_ml.parquet \
  --output-model models/wot_to_dota_anomaly.joblib \
  --output-report reports/wot_to_dota_anomaly.json \
  --normal-label 0 \
  --use-custom-stopwords
```

For fixed project presets, use [src/run_anomaly_presets.py](/Users/paolacalle/Desktop/NYU/semesters/spring-2026/ML/hw/gaming-toxicity-detection/src/run_anomaly_presets.py:1):

```bash
python src/run_anomaly_presets.py wot_to_dota
python src/run_anomaly_presets.py dota_to_wot
```

Available presets:
- `wot_to_wot`
- `wot_to_dota`
- `dota_to_dota`
- `dota_to_wot`
