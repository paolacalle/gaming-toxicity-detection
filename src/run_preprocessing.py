#!/usr/bin/env python
"""
Gaming toxicity preprocessing pipeline.

Ingests raw CSVs, runs per-step cleaning with CV evaluation,
outputs cleaned parquet (message + label only) + ablation CSV.

Output
------
  <output-dir>/<name>.parquet                   fully cleaned dataset
  <output-dir>/cleaning_ablation_<name>.csv     per-step CV metrics

Usage (PowerShell)
------------------
# WoT
python -m src.run_preprocessing `
    --preset wot `
    --name WoT `
    --inputs data/raw_data/wot/train.csv data/raw_data/wot/val.csv `
    --test-text data/raw_data/wot/test_index_text.csv `
    --test-label data/raw_data/wot/test_index_label.csv `
    --output-dir data/processed_data/wot

# Dota
python -m src.run_preprocessing `
    --preset dota `
    --name Dota `
    --inputs data/raw_data/dota/CONDA_train.csv data/raw_data/dota/CONDA_valid.csv `
    --output-dir data/processed_data/dota
"""

import argparse
import html as html_lib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from lingua import Language, LanguageDetectorBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tokenizer import tokenize
from src.prep_evaluation_multi import eval_step

# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
SEED = 7524
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
NON_LATIN_SCRIPT = re.compile(
    r"[\u0400-\u04FF"   # Cyrillic
    r"\u4E00-\u9FFF"    # CJK unified ideographs
    r"\u3400-\u4DBF"    # CJK extension A
    r"\uF900-\uFAFF"    # CJK compatibility ideographs
    r"\u0600-\u06FF"    # Arabic
    r"\u0590-\u05FF"    # Hebrew
    r"\u3040-\u30FF"    # Japanese (Hiragana + Katakana)
    r"\uAC00-\uD7AF"    # Korean (Hangul syllables)
    r"\u1100-\u11FF"    # Korean (Hangul Jamo)
    r"\u0E00-\u0E7F"    # Thai
    r"\u0900-\u097F"    # Devanagari (Hindi)
    r"\u0980-\u09FF"    # Bengali
    r"\u0370-\u03FF"    # Greek
    r"\u10A0-\u10FF"    # Georgian
    r"\u0530-\u058F"    # Armenian
    r"\u1000-\u109F"    # Myanmar
    r"\u1780-\u17FF]"   # Khmer
)

# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------
PRESETS = {
    "wot": {
        "steps": ["baseline", "non_latin", "non_english", "majority_map",
                  "artifacts", "label_fix", "svc_check"],
        "artifact_error": True,
        "artifact_html":  True,
        "svc_balanced":   True,
    },
    "dota": {
        "steps": ["baseline", "non_latin", "non_english", "sepa_strip",
                  "majority_map", "label_fix", "svc_check"],
        "message_col":  "utterance",
        "label_col":    "intentClass",
        "artifact_error": False,
        "artifact_html":  False,
        "svc_balanced":   True,
    },
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_detector():
    return (
        LanguageDetectorBuilder
        .from_all_languages_with_latin_script()
        .with_minimum_relative_distance(0.25)
        .build()
    )


def _is_latin_non_english(text, detector, min_len=30):
    if len(str(text).strip()) < min_len:
        return False
    lang = detector.detect_language_of(str(text))
    return lang is not None and lang != Language.ENGLISH


def _read_file(path: Path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_data(args, preset):
    dfs = []
    for p in args.inputs:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        dfs.append(_read_file(p))

    # Optional test split merge (WoT)
    if args.test_text and args.test_label:
        text_path, label_path = Path(args.test_text), Path(args.test_label)
        if not text_path.exists():
            raise FileNotFoundError(f"Test text file not found: {text_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Test label file not found: {label_path}")
        test = _read_file(text_path).merge(
            _read_file(label_path), on=args.test_merge_on, how="inner"
        )
        dfs.append(test)

    df = pd.concat(dfs, ignore_index=True)

    # Rename to canonical message / label
    msg_col   = preset.get("message_col", "message")
    label_col = preset.get("label_col",   "label")
    rename = {}
    if msg_col != "message":
        rename[msg_col] = "message"
    if label_col != "label":
        rename[label_col] = "label"
    if rename:
        df = df.rename(columns=rename)

    for col in ("message", "label"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    df = df.dropna(subset=["message"]).reset_index(drop=True)

    # Label encoding — auto-detect if non-numeric
    if not pd.api.types.is_integer_dtype(df["label"]) and df["label"].dtype == object:
        unique_vals = df["label"].dropna().drop_duplicates().tolist()
        auto_map = {v: i for i, v in enumerate(unique_vals)}
        print(f"Auto label map: {auto_map}")
        df["label"] = df["label"].map(auto_map)
        df = df.dropna(subset=["label"]).reset_index(drop=True)

    df["label"] = df["label"].astype(int)

    # Keep only what downstream needs
    return df[["message", "label"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# helpers shared by steps
# ---------------------------------------------------------------------------

def _dataset_path(args):
    return Path(args.output_dir) / f"{args.name.lower()}.parquet"


def _save_and_eval(df, step_name, accumulator, args, clf=None):
    out = _dataset_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return eval_step(
        f"{args.name.lower()}_{step_name}",
        preprocessing_impact=accumulator,
        datasets=(args.name,),
        clf=clf,
        dataset_paths={args.name: out},
    )


# ---------------------------------------------------------------------------
# preprocessing steps
# ---------------------------------------------------------------------------

def step_baseline(df, args, preset, accumulator, step_name):
    print(f"\n=== Baseline ({args.name}) — rows: {len(df)} ===")
    return df, _save_and_eval(df, step_name, None, args)


def step_non_latin(df, args, preset, accumulator, step_name):
    print(f"\n=== Non-Latin script drop ({args.name}) ===")
    before = len(df)
    df = df[~df["message"].str.contains(NON_LATIN_SCRIPT, regex=True, na=False)].reset_index(drop=True)
    dropped = before - len(df)
    print(f"Dropped {dropped} ({dropped / before:.1%}) | Remaining: {len(df)}")
    return df, _save_and_eval(df, step_name, accumulator, args)


def step_non_english(df, args, preset, accumulator, step_name):
    print(f"\n=== Non-English (Latin) drop ({args.name}) ===")
    detector = _build_detector()
    mask = df["message"].apply(lambda t: _is_latin_non_english(t, detector))
    before = len(df)
    df = df[~mask].reset_index(drop=True)
    dropped = before - len(df)
    print(f"Dropped {dropped} ({dropped / before:.1%}) | Remaining: {len(df)}")
    return df, _save_and_eval(df, step_name, accumulator, args)


def step_sepa_strip(df, args, preset, accumulator, step_name):
    print(f"\n=== SEPA strip ({args.name}) ===")
    df = df.copy()
    df["message"] = df["message"].str.replace(r"\s*\[SEPA\]\s*", " ", regex=True).str.strip()
    remaining = df["message"].str.contains(r"\[SEPA\]", regex=True).sum()
    print(f"[SEPA] remaining: {remaining}")
    return df, _save_and_eval(df, step_name, accumulator, args)


def step_majority_map(df, args, preset, accumulator, step_name):
    print(f"\n=== Majority-label dedup ({args.name}) ===")
    conflicts = df.groupby("message")["label"].nunique()
    conflicts = conflicts[conflicts > 1]
    majority = df.groupby("message")["label"].agg(lambda x: x.value_counts().index[0])
    df = df.copy()
    df["label"] = df["message"].map(majority)
    if len(conflicts):
        print(f"Resolved {len(conflicts)} conflicting messages via majority vote")
    return df, _save_and_eval(df, step_name, accumulator, args)


def step_artifacts(df, args, preset, accumulator, step_name):
    print(f"\n=== Artifact removal ({args.name}) ===")
    if preset.get("artifact_error"):
        mask = df["message"].str.contains(r"#ERROR!", regex=False, na=False)
        print(f"Dropping #ERROR! rows: {mask.sum()}")
        df = df[~mask].reset_index(drop=True)
    if preset.get("artifact_html"):
        html_mask = df["message"].str.contains(r"&\w+;", regex=True, na=False)
        print(f"HTML entities before decode: {html_mask.sum()}")
        df = df.copy()
        df["message"] = df["message"].apply(html_lib.unescape)
        print(f"HTML entities after decode: {df['message'].str.contains(r'&\w+;', regex=True, na=False).sum()}")
    print(f"Shape: {df.shape}")
    return df, _save_and_eval(df, step_name, accumulator, args)


def step_label_fix(df, args, preset, accumulator, step_name):
    print(f"\n=== Label fix with cleanlab ({args.name}) ===")
    X, y = df["message"].values, df["label"].astype(int).values
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.95,
            sublinear_tf=True, norm="l2",
            analyzer="word", tokenizer=tokenize, token_pattern=None,
        )),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")),
    ])
    oof_probs = cross_val_predict(pipe, X, y, cv=5, method="predict_proba", n_jobs=-1)
    issue_idx = find_label_issues(labels=y, pred_probs=oof_probs,
                                  return_indices_ranked_by="self_confidence")
    print(f"Suspected mislabeled: {len(issue_idx)} ({len(issue_idx) / len(y):.1%})")
    if len(issue_idx):
        print(df.iloc[issue_idx[:50]][["message", "label"]].assign(
            predicted=oof_probs[issue_idx[:50]].argmax(axis=1)
        ).to_string())
    df = df.copy()
    df.loc[issue_idx, "label"] = oof_probs[issue_idx].argmax(axis=1)
    print(f"Relabeled {len(issue_idx)} rows")
    return df, _save_and_eval(df, step_name, accumulator, args)


def step_svc_check(df, args, preset, accumulator, step_name):
    print(f"\n=== SVC circularity check ({args.name}) ===")
    svc_kwargs = dict(max_iter=2000, random_state=SEED)
    if preset.get("svc_balanced"):
        svc_kwargs["class_weight"] = "balanced"
    out = _dataset_path(args)
    acc = eval_step(
        f"{args.name.lower()}_{step_name}",
        preprocessing_impact=accumulator,
        datasets=(args.name,),
        clf=LinearSVC(**svc_kwargs),
        dataset_paths={args.name: out},
    )
    return df, acc


STEP_REGISTRY = {
    "baseline":    step_baseline,
    "non_latin":   step_non_latin,
    "non_english": step_non_english,
    "sepa_strip":  step_sepa_strip,
    "majority_map": step_majority_map,
    "artifacts":   step_artifacts,
    "label_fix":   step_label_fix,
    "svc_check":   step_svc_check,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaming toxicity preprocessing pipeline"
    )
    parser.add_argument("--preset", required=True,
                        choices=list(PRESETS.keys()),
                        help="Dataset preset: " + ", ".join(PRESETS.keys()))
    parser.add_argument("--name", required=True,
                        help="Dataset name used in output filenames and step labels")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input CSV/Parquet files to concatenate")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write parquet and ablation CSV")
    parser.add_argument("--results-csv", default=None,
                        help="Override ablation CSV path (default: <output-dir>/cleaning_ablation_<name>.csv)")

    # WoT test split
    parser.add_argument("--test-text", default=None,
                        help="Test text CSV (WoT only)")
    parser.add_argument("--test-label", default=None,
                        help="Test label CSV (WoT only)")
    parser.add_argument("--test-merge-on", default="index",
                        help="Key column for test text/label merge (default: index)")

    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    steps  = preset["steps"]

    df = load_data(args, preset)
    print(f"\nLoaded '{args.name}': {df.shape}")

    accumulator = None
    for sname in steps:
        df, accumulator = STEP_REGISTRY[sname](df, args, preset, accumulator, step_name=sname)

    results_path = (
        Path(args.results_csv) if args.results_csv
        else Path(args.output_dir) / f"cleaning_ablation_{args.name.lower()}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    accumulator.to_csv(results_path, index=False)
    print(f"\nSaved ablation results → {results_path}")

    print("\n=== Final dataset ===")
    print(f"Shape: {df.shape}")
    counts = df["label"].value_counts().sort_index()
    pct    = df["label"].value_counts(normalize=True).sort_index().mul(100).round(1)
    print(pd.DataFrame({"count": counts, "pct%": pct}).to_string())


if __name__ == "__main__":
    main()
