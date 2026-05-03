"""
data.py — Dataset loading, preprocessing, splitting, and PyTorch Dataset wrapper.

Public API
----------
load_dataframe(path)                 → pd.DataFrame
light_clean(text)                    → str
make_three_way_split(df, ...)        → (train_df, val_df, test_df)
prepare_labels(df, label_col, task)  → list[int]
get_texts(df, text_col, clean_col)   → list[str]
ToxicityDataset                      → torch.utils.data.Dataset
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def light_clean(text: str) -> str:
    """
    Minimal normalisation for BERT inputs.

    BERT's WordPiece tokenizer works best on natural, lower-cased text.
    We deliberately avoid stemming or stopword removal — those strip the
    contextual signals that transformer attention relies on.

    Steps:
        1. Cast to string and strip leading/trailing whitespace.
        2. Lowercase.
        3. Collapse any run of whitespace (spaces, tabs, newlines) to a
           single space.
    """
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def get_texts(
    df: pd.DataFrame,
    text_col: str = "message",
    clean_col: str = "clean_message",
) -> list:
    """
    Return the best available text string for every row.

    Preference order:
        1. ``clean_col`` — if the column exists and the value is non-empty.
        2. ``text_col``  — raw original message, used as a fallback so no
           row ever ends up with an empty BERT input.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    text_col : str
        Name of the raw-text column (default ``"message"``).
    clean_col : str
        Name of the pre-cleaned column (default ``"clean_message"``).

    Returns
    -------
    list[str]
        One string per row, guaranteed non-null.
    """
    if clean_col in df.columns:
        return (
            df.apply(
                lambda r: r[clean_col]
                if pd.notna(r[clean_col]) and str(r[clean_col]).strip()
                else r[text_col],
                axis=1,
            )
            .fillna("")
            .astype(str)
            .tolist()
        )
    # clean_col not present — use raw text directly
    return df[text_col].fillna("").astype(str).tolist()


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def prepare_labels(
    df: pd.DataFrame,
    label_col: str = "label",
    task_type: str = "binary",
) -> list:
    """
    Convert a DataFrame label column to a flat Python list of integers.

    Binary mode:  any value > 0 becomes 1 (toxic); 0 stays 0 (non-toxic).
                  This makes cross-domain comparison feasible when datasets
                  use different multi-class scales.
    Multiclass:   labels are cast to int and used as class indices directly.
                  The caller is responsible for ensuring they are 0-based
                  and contiguous.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
    task_type : str
        ``"binary"`` or ``"multiclass"``.

    Returns
    -------
    list[int]
    """
    if task_type == "binary":
        return (df[label_col] > 0).astype(int).tolist()
    elif task_type == "multiclass":
        return df[label_col].astype(int).tolist()
    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Choose 'binary' or 'multiclass'."
        )


def infer_num_classes(labels: list, task_type: str) -> int:
    """
    Infer the number of output classes from a label list.

    For binary tasks this always returns 2.
    For multiclass it returns ``max(labels) + 1``, which assumes 0-based
    contiguous integer labels.

    Parameters
    ----------
    labels : list[int]
    task_type : str

    Returns
    -------
    int
    """
    if task_type == "binary":
        return 2
    return int(max(labels)) + 1


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a ``.csv`` or ``.parquet`` file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the file extension is not ``.csv`` or ``.parquet``.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Use '.csv' or '.parquet'."
        )


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def make_three_way_split(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Create a stratified 70 / 15 / 15 train / val / test split.

    Stratification key
    ------------------
    We stratify on the **binary** version of the label (any label > 0 = toxic)
    rather than the full multi-class label.  This prevents
    ``train_test_split`` from raising a ``ValueError`` when a rare class
    (e.g. WOT toxicity level 5, which has ~20 total examples) would end up
    with fewer than 2 samples in one split.

    Split sizes
    -----------
    Given *N* total rows:
        - test  = floor(N × test_size)
        - val   = floor(remaining × val_size / (1 − test_size))
        - train = the rest

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Column used to derive the stratification key.
    test_size : float
        Fraction of the full dataset reserved for testing.
    val_size : float
        Fraction of the full dataset reserved for validation.
    seed : int

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) — each with a fresh RangeIndex.
    """
    binary_strat = (df[label_col] > 0).astype(int)

    # Step 1 — carve out the held-out test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=binary_strat,
    )

    # Step 2 — split the remaining data into train and val
    # val_size / (1 - test_size) converts the target fraction from
    # "fraction of total" to "fraction of the train_val pool".
    binary_strat_tv = (train_val[label_col] > 0).astype(int)
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=binary_strat_tv,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def preprocess_and_split(
    raw_path: str,
    output_dir: str,
    domain: str,
    text_col: str = "message",
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Load a raw parquet/CSV, add preprocessing columns, apply a 70/15/15
    split, and save the three parquets to *output_dir*.

    Added columns
    -------------
    ``comment_length``  — word count of the original message.
    ``clean_message``   — lower-cased, whitespace-normalised text
                          (good default input for BERT).

    Parameters
    ----------
    raw_path : str
        Path to the unsplit source file.
    output_dir : str
        Destination directory for ``{domain}_train/val/test.parquet``.
    domain : str
        Short name used to construct output filenames (e.g. ``"wot"``).
    text_col, label_col, test_size, val_size, seed
        Forwarded to helpers.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) — also persisted to disk.
    """
    df = load_dataframe(raw_path)

    # Add derived columns used by both BERT and traditional ML methods
    df["comment_length"] = df[text_col].str.split().str.len()
    df["clean_message"] = df[text_col].apply(light_clean)

    train_df, val_df, test_df = make_three_way_split(
        df, label_col=label_col, test_size=test_size, val_size=val_size, seed=seed
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [text_col, label_col, "comment_length", "clean_message"]
    train_df[keep_cols].to_parquet(output_dir / f"{domain}_train.parquet", index=False)
    val_df[keep_cols].to_parquet(output_dir / f"{domain}_val.parquet",   index=False)
    test_df[keep_cols].to_parquet(output_dir / f"{domain}_test.parquet",  index=False)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ToxicityDataset:
    """
    Tokenises a list of texts up-front and stores integer labels.

    Implements the ``torch.utils.data.Dataset`` interface (``__len__`` and
    ``__getitem__``) without importing torch at module level, so the
    preprocessing utilities in this file can be used independently of a
    PyTorch installation.

    All sequences are padded / truncated to exactly ``max_length`` tokens so
    that samples can be stacked into fixed-size batches without a custom
    collate function.

    Parameters
    ----------
    texts : list[str]
        Raw text inputs (one per sample).
    labels : list[int]
        Integer class labels aligned with *texts*.
    tokenizer
        Any HuggingFace fast or slow tokenizer.
    max_length : int
        Token budget per sequence.  Use 64 for short game-chat messages
        and 256 for longer social-media text.

    Examples
    --------
    >>> from transformers import BertTokenizer
    >>> tok = BertTokenizer.from_pretrained("bert-base-uncased")
    >>> ds  = ToxicityDataset(["gg noob", "well played"], [1, 0], tok, 32)
    >>> len(ds)
    2
    >>> ds[0]["input_ids"].shape
    torch.Size([32])
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 64,
    ):
        # Defer the torch import to construction time so that the rest of the
        # module (light_clean, get_texts, make_three_way_split, …) can be
        # imported without a PyTorch installation being present.
        import torch
        from torch.utils.data import Dataset as _Dataset

        # Tokenise all texts at once — this is faster than tokenising per
        # __getitem__ call and avoids re-loading the tokenizer model.
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Return a dict of tensors: input_ids, attention_mask,
        # (token_type_ids if present), and labels.
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
