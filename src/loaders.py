from pathlib import Path
import pandas as pd
from src.label_schemes import apply_scheme

# Anchor paths from src/ location — works from any caller depth (notebook or script)
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
