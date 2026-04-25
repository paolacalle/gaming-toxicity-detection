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
