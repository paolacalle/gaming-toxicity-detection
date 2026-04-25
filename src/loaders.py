from pathlib import Path
import pandas as pd

# Anchor paths from src/ location — works from any caller depth
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_WOT  = PROJECT_ROOT / "data" / "processed_data" / "wot"
DATA_DIR_DOTA = PROJECT_ROOT / "data" / "processed_data" / "dota"

_WOT_FILES  = {'train': 'wot_train_ml.parquet', 'val': 'wot_val_ml.parquet'}
_DOTA_FILES = {'train': 'dota_train_ml.parquet', 'val': 'dota_val_ml.parquet'}


def load_wot(split: str) -> pd.DataFrame:
    """Load WoT split. Returns raw labels (0-5)."""
    if split not in _WOT_FILES:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")
    return pd.read_parquet(DATA_DIR_WOT / _WOT_FILES[split])


def load_dota(split: str) -> pd.DataFrame:
    """Load Dota split. Returns raw labels (0-3)."""
    if split not in _DOTA_FILES:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")
    return pd.read_parquet(DATA_DIR_DOTA / _DOTA_FILES[split])


def load_combined(split: str) -> pd.DataFrame:
    """Concatenate WoT + Dota splits with raw labels."""
    wot  = load_wot(split)
    dota = load_dota(split)
    return pd.concat([wot, dota], ignore_index=True)
