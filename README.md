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
