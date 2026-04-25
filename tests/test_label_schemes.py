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
