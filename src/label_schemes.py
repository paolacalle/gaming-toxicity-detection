import pandas as pd

# WoT: 6 classes — 0=Non-Toxic, 1=Insults, 2=Other Offensive,
#                   3=Hate, 4=Threats, 5=Extremism
# Incremental order: severity-based. Easiest discrimination first.
# Mild = Other Offensive + Hate (broad, less targeted)
# Severe = Insults + Threats + Extremism (direct harm, evasion)
WOT_SCHEMES: dict[int, dict[int, int]] = {
    2: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
    3: {0: 0, 2: 1, 3: 1, 1: 2, 4: 2, 5: 2},
    4: {0: 0, 2: 1, 3: 2, 1: 3, 4: 3, 5: 3},
    5: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4},
    6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
}

# WoT class names per n_classes for display
WOT_CLASS_NAMES: dict[int, list[str]] = {
    2: ['Non-Toxic', 'Toxic'],
    3: ['Non-Toxic', 'Mild', 'Severe'],
    4: ['Non-Toxic', 'Other Offensive', 'Hate', 'Threats+Insults+Extremism'],
    5: ['Non-Toxic', 'Insults', 'Other Offensive', 'Hate', 'Threats+Extremism'],
    6: ['Non-Toxic', 'Insults', 'Other Offensive', 'Hate', 'Threats', 'Extremism'],
}

# Dota: 4 classes — 0=Other/Non-Toxic, 1=Ego, 2=Aggression, 3=Impolite
# Incremental order: Impolite first (condescension, mild), then Ego+Aggression
DOTA_SCHEMES: dict[int, dict[int, int]] = {
    2: {0: 0, 1: 1, 2: 1, 3: 1},
    3: {0: 0, 3: 1, 1: 2, 2: 2},
    4: {0: 0, 1: 1, 2: 2, 3: 3},
}

DOTA_CLASS_NAMES: dict[int, list[str]] = {
    2: ['Non-Toxic', 'Toxic'],
    3: ['Non-Toxic', 'Impolite', 'Ego+Aggression'],
    4: ['Non-Toxic', 'Ego', 'Aggression', 'Impolite'],
}


def apply_scheme(series: pd.Series, scheme: dict[int, int]) -> pd.Series:
    """Map original labels to new label space. Raises KeyError on unknown label."""
    result = series.map(scheme)
    if result.isna().any():
        raise KeyError("Unknown label in series")
    return result
