"""
data.py — Dataset loading, preprocessing, splitting, and PyTorch Dataset wrapper.

Public API
----------
load_dataframe(path)              → pd.DataFrame
light_clean(text)                 → str
make_splits(df, ...)              → (train_df, test_df)
apply_label_scheme(df, scheme)    → pd.DataFrame  (label column remapped)
prepare_labels(df, ...)           → list[int]
get_texts(df, ...)                → list[str]
ToxicityDataset                   → torch.utils.data.Dataset

Label schemes
-------------
All three datasets are remapped to a shared 3-class space before training:
    0 — Non-Toxic
    1 — Mild toxicity  (insults, low-level offence)
    2 — Severe toxicity (hate, threats, extremism, identity attacks)

WOT_SCHEME_3  : WOT raw labels 0-5  → {0→0, 1→1, 2→1, 3→2, 4→2, 5→2}
DOTA_SCHEME_3 : DOTA raw labels 0-3 → {0→0, 1→2, 2→2, 3→1}
jigsaw3       : derived from Jigsaw sub-label columns:
                  severe (2) if any of severe_toxic/obscene/threat/identity_hate
                  mild   (1) if insult only
                  non-toxic (0) otherwise
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Label scheme constants
# ---------------------------------------------------------------------------

# n=3: Non-Toxic / Insults+OtherOffensive (Mild) / Hate+Threats+Extremism (Severe)
WOT_SCHEME_3: dict = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}

# n=3: Non-Toxic / Ego+Aggression (Mild) / Impolite (Severe)
DOTA_SCHEME_3: dict = {0: 0, 1: 2, 2: 2, 3: 1}

# Registry for dict-based schemes
_SCHEME_REGISTRY: dict = {
    "wot3":  WOT_SCHEME_3,
    "dota3": DOTA_SCHEME_3,
}


# ---------------------------------------------------------------------------
# Label scheme application
# ---------------------------------------------------------------------------

def apply_label_scheme(
    df: pd.DataFrame,
    scheme: str,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Remap the integer label column to a standardised 3-class space.

    Supported schemes
    -----------------
    ``"wot3"``
        Maps WOT raw labels 0–5 to {0→Non-Toxic, 1→Mild, 2→Severe}
        using :data:`WOT_SCHEME_3`.

    ``"dota3"``
        Maps DOTA raw labels 0–3 to {0→Non-Toxic, 1→Mild, 2→Severe}
        using :data:`DOTA_SCHEME_3`.

    ``"jigsaw3"``
        Derives 3-class labels from the six original Jigsaw sub-label
        columns that must be present in *df*:

        ==================  ====
        Condition           Class
        ==================  ====
        severe_toxic=1,     2 (Severe)
        obscene=1,
        threat=1, or
        identity_hate=1
        insult=1 (only)     1 (Mild)
        none active         0 (Non-Toxic)
        ==================  ====

        Priority: severe wins over mild.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose *label_col* will be overwritten with remapped values.
        Must contain the Jigsaw sub-label columns when ``scheme="jigsaw3"``.
    scheme : str
        One of ``"wot3"``, ``"dota3"``, or ``"jigsaw3"``.
    label_col : str
        Name of the column to write remapped labels into.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with *label_col* replaced by the remapped integers.

    Raises
    ------
    ValueError
        If the scheme is unknown, required columns are missing, or the
        mapping leaves any labels as NaN.
    """
    df = df.copy()

    if scheme == "jigsaw3":
        _SEVERE_COLS = ["severe_toxic", "obscene", "threat", "identity_hate"]
        _MILD_COL    = "insult"
        missing = (set(_SEVERE_COLS) | {_MILD_COL}) - set(df.columns)
        if missing:
            raise ValueError(
                f"'jigsaw3' scheme requires columns {sorted(missing)} "
                f"— not found in DataFrame."
            )
        severe = (
            df["severe_toxic"].astype(bool)
            | df["obscene"].astype(bool)
            | df["threat"].astype(bool)
            | df["identity_hate"].astype(bool)
        )
        mild = df[_MILD_COL].astype(bool) & ~severe

        new_labels = pd.Series(0, index=df.index, dtype=int)
        new_labels[mild]   = 1
        new_labels[severe] = 2
        df[label_col] = new_labels
        return df

    mapping = _SCHEME_REGISTRY.get(scheme)
    if mapping is None:
        raise ValueError(
            f"Unknown label scheme '{scheme}'. "
            f"Available: {sorted(_SCHEME_REGISTRY)} + ['jigsaw3']"
        )

    df[label_col] = df[label_col].astype(int).map(mapping)
    if df[label_col].isna().any():
        bad = sorted(df.loc[df[label_col].isna()].index[:5].tolist())
        raise ValueError(
            f"Scheme '{scheme}' left {df[label_col].isna().sum()} "
            f"unmapped rows (sample indices: {bad}). "
            "Check that the mapping covers all raw label values."
        )
    df[label_col] = df[label_col].astype(int)
    return df


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

def make_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Stratified two-way train / test split.

    Validation is not produced here because cross-validation handles the
    train-time validation signal internally (each fold creates its own
    held-out fold slice).  This function is used only to carve a final
    held-out test set that is never seen during training or CV.

    Stratification
    --------------
    Stratifies on the **binary** version of the label (``label > 0``)
    rather than the raw multi-class label.  This prevents
    ``train_test_split`` from raising a ``ValueError`` when a rare class
    (e.g. WOT toxicity level 5 with ~20 total examples) would end up with
    fewer than two samples in one split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset to split (used in ``--dataset`` / single-file mode).
    label_col : str
        Integer label column used to derive the binary stratification key.
    test_size : float
        Fraction reserved for the held-out test set (default 0.15 → 85/15).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` — each with a fresh RangeIndex.
    """
    binary_strat = (df[label_col] > 0).astype(int)

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=binary_strat,
    )

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def preprocess_and_split(
    raw_path: str,
    output_dir: str,
    domain: str,
    text_col: str = "message",
    label_col: str = "label",
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Load a raw parquet/CSV, add preprocessing columns, apply an 85/15
    train/test split, and save both parquets to *output_dir*.

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
        Destination directory for ``{domain}_train.parquet`` and
        ``{domain}_test.parquet``.
    domain : str
        Short name used to construct output filenames (e.g. ``"wot"``).
    text_col, label_col, test_size, seed
        Forwarded to helpers.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) — also persisted to disk.
    """
    df = load_dataframe(raw_path)

    # Add derived columns used by both BERT and traditional ML methods
    df["comment_length"] = df[text_col].str.split().str.len()
    df["clean_message"] = df[text_col].apply(light_clean)

    train_df, test_df = make_splits(
        df, label_col=label_col, test_size=test_size, seed=seed
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [text_col, label_col, "comment_length", "clean_message"]
    train_df[keep_cols].to_parquet(output_dir / f"{domain}_train.parquet", index=False)
    test_df[keep_cols].to_parquet( output_dir / f"{domain}_test.parquet",  index=False)

    return train_df, test_df


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
