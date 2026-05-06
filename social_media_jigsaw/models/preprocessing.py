"""
preprocessing.py — shared text cleaning pipeline for toxicity detection.

Two variants:
  - clean_text(text)              : standard cleaning (lowercase, URLs, HTML, punct)
  - clean_text(text, slang=True)  : additionally normalises slang using the
                                    MLBtrio/genz-slang-dataset from HuggingFace

Usage:
    from preprocessing import clean_text, preprocess_df
    train['clean'] = preprocess_df(train['comment_text'])
    train['clean_slang'] = preprocess_df(train['comment_text'], slang=True)
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Domain stopwords — Wikipedia platform terms that are dataset artifacts.
# EDA showed these dominate top-token charts without carrying toxicity signal.
# Export this so downstream models (TF-IDF vectorizer) can use it directly.
# ---------------------------------------------------------------------------
DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    'wikipedia', 'wiki', 'article', 'articles', 'talk', 'page', 'pages',
    'edit', 'edits', 'edited', 'editing', 'editor', 'editors',
    'user', 'users', 'username', 'block', 'blocked', 'blocking',
    'template', 'unsigned', 'utc', 'http', 'www', 'com',
    'archive', 'revision', 'section', 'category', 'source',
    'deleted', 'deletion', 'admin', 'administrator',
})

# ---------------------------------------------------------------------------
# Slang map — loaded lazily from HuggingFace on first use
# ---------------------------------------------------------------------------
_SLANG_MAP: dict[str, str] | None = None          # single-token slang
_PHRASE_PATTERNS: list[tuple] | None = None        # (compiled_re, replacement)


def _first_sentence(text: str) -> str:
    """Return the first sentence of a description, stripped and lowercased."""
    text = text.strip()
    # Cut at first period that ends a clause (not mid-abbreviation)
    match = re.search(r'\.(?:\s|$)', text)
    if match:
        text = text[:match.start()]
    return text.lower().strip()


def _load_slang_map() -> tuple[dict[str, str], list[tuple]]:
    """
    Load MLBtrio/genz-slang-dataset and build:
      - token_map   : {slang_lower: description} for single-word slang
      - phrase_pats : [(compiled_regex, replacement)] for multi-word slang
    Sorted longest-phrase-first so longer matches take priority.

    Two guard rails applied when building the map:
      1. First-match-wins — the dataset has 170 duplicate slang terms with
         conflicting descriptions (e.g. GG appears 4 times; last entry
         "Brother" would overwrite the correct "Good Game"). We keep only the
         first description seen for each normalised key.
      2. Common-word filter — some raw entries contain punctuation that gets
         stripped during normalisation, producing keys that are ordinary English
         words (e.g. "A/N" → "an", "WAS" → "was"). Replacing those tokens
         in every sentence would corrupt the text, so we skip any key that
         appears in sklearn's English stopword list.
    """
    from datasets import load_dataset
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    ds = load_dataset("MLBtrio/genz-slang-dataset", split="train")
    df = ds.to_pandas()

    token_map: dict[str, str] = {}
    seen_phrases: set[str] = set()
    phrases: list[tuple[str, str]] = []  # (slang_lower, description)

    for _, row in df.iterrows():
        slang = str(row["Slang"]).strip()
        desc  = _first_sentence(str(row["Description"]))
        if not slang or not desc:
            continue

        # Normalise key: lowercase, strip punctuation (keep spaces for phrases)
        key = re.sub(r"[^a-z0-9\s]", "", slang.lower()).strip()
        if not key:
            continue

        # Guard 2: skip if normalised key is a common English word.
        # This catches cases like "A/N" → "an" and "WAS" → "was".
        if key in ENGLISH_STOP_WORDS:
            continue

        if " " in key:
            # Guard 1 (phrases): skip duplicate phrase entries
            if key not in seen_phrases:
                seen_phrases.add(key)
                phrases.append((key, desc))
        else:
            # Guard 1 (tokens): first-match-wins — do not overwrite
            if key not in token_map:
                token_map[key] = desc

    # Sort phrases longest-first so "catch these hands" beats "catch"
    phrases.sort(key=lambda x: len(x[0]), reverse=True)
    phrase_pats = [
        (re.compile(r'\b' + re.escape(p) + r'\b'), rep)
        for p, rep in phrases
    ]

    return token_map, phrase_pats


def _get_slang_resources() -> tuple[dict[str, str], list[tuple]]:
    """Return cached slang resources, loading them on first call."""
    global _SLANG_MAP, _PHRASE_PATTERNS
    if _SLANG_MAP is None:
        print("[preprocessing] Loading slang dataset from HuggingFace...")
        _SLANG_MAP, _PHRASE_PATTERNS = _load_slang_map()
        print(f"[preprocessing] Loaded {len(_SLANG_MAP)} single-token and "
              f"{len(_PHRASE_PATTERNS)} multi-word slang entries.")
    return _SLANG_MAP, _PHRASE_PATTERNS


# ---------------------------------------------------------------------------
# Compiled regex patterns for standard cleaning
# ---------------------------------------------------------------------------
_RE_URL        = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML       = re.compile(r'<[^>]+>')
_RE_NEWLINE    = re.compile(r'[\r\n\t]+')
_RE_REPEATED   = re.compile(r'(.)\1{3,}')       # "fuuuuck" → "fuuck" (keep 2)
_RE_PUNCT      = re.compile(r"[^\w\s']")         # strip punct but keep apostrophes
_RE_STRAY_APOS = re.compile(r"(?<!\w)'|'(?!\w)") # remove apostrophes not in contractions
_RE_SPACES     = re.compile(r'\s{2,}')


def _apply_slang_normalization(text: str) -> str:
    """
    Apply phrase-level then token-level slang normalization.
    Must be called on already-lowercased, punctuation-stripped text.
    """
    token_map, phrase_pats = _get_slang_resources()

    # 1. Multi-word phrases first (longest match wins due to sort order)
    for pattern, replacement in phrase_pats:
        text = pattern.sub(replacement, text)

    # 2. Single-token replacement
    tokens = text.split()
    tokens = [token_map.get(tok, tok) for tok in tokens]
    return ' '.join(tokens)


def clean_text(text: str, slang: bool = False) -> str:
    """
    Clean a single comment string.

    Parameters
    ----------
    text  : raw comment string
    slang : if True, expand slang terms using MLBtrio/genz-slang-dataset

    Returns
    -------
    cleaned string
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _RE_URL.sub(' ', text)
    text = _RE_HTML.sub(' ', text)
    text = _RE_NEWLINE.sub(' ', text)
    text = _RE_REPEATED.sub(r'\1\1', text)
    text = _RE_PUNCT.sub(' ', text)
    text = _RE_STRAY_APOS.sub(' ', text)
    text = _RE_SPACES.sub(' ', text).strip()

    if slang:
        text = _apply_slang_normalization(text)
        text = _RE_SPACES.sub(' ', text).strip()

    return text


def preprocess_df(series: pd.Series, slang: bool = False) -> pd.Series:
    """
    Apply clean_text to a pandas Series.

    Parameters
    ----------
    series : pd.Series of raw comment strings
    slang  : passed through to clean_text

    Returns
    -------
    pd.Series of cleaned strings
    """
    return series.apply(lambda x: clean_text(x, slang=slang))


def extract_length_features(series: pd.Series) -> pd.DataFrame:
    """
    Extract character length and word count for each comment.

    EDA finding: toxic comments are shorter on average (123 median words vs 216
    for non-toxic), so these are informative features alongside TF-IDF.

    Parameters
    ----------
    series : pd.Series of raw or cleaned comment strings

    Returns
    -------
    pd.DataFrame with columns ['char_len', 'word_count']
    """
    return pd.DataFrame({
        'char_len':   series.str.len().fillna(0).astype(float),
        'word_count': series.str.split().str.len().fillna(0).astype(float),
    })


def slang_density(text: str) -> float:
    """
    Return the fraction of tokens in `text` that are known slang terms.
    Useful as a feature or for exploratory analysis.
    Triggers slang dataset load on first call.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    token_map, _ = _get_slang_resources()
    tokens = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in token_map)
    return hits / len(tokens)


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    examples = [
        "STFU u absolute noob lmaooo!!!",
        "Check this out: https://example.com wtf is this??",
        "<b>You're</b> an idiot kys",
        "Good game everyone, gg wp!!!!!",
        "That play was lowkey bussin fr no cap",
        "Catch these hands you absolute stan",
        "My G you tripping, that's straight up gas.",
        "wth is this I don't wanna be here smh"
    ]

    print(f"{'RAW':<50} | {'CLEANED':<45} | CLEANED+SLANG")
    print('-' * 160)
    for ex in examples:
        print(f"{ex:<50} | {clean_text(ex):<45} | {clean_text(ex, slang=True)}")

    print()
    print('Slang density:', [round(slang_density(e), 3) for e in examples])
