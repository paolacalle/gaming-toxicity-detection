import re
import pandas as pd


class SocialMediaTextPreprocessor:
    """
    Text preprocessing pipeline for toxicity detection.

    Supports:
    - standard cleaning
    - optional slang normalization
    - length feature extraction
    - slang density calculation
    """

    DOMAIN_STOPWORDS: frozenset[str] = frozenset({
        "wikipedia", "wiki", "article", "articles", "talk", "page", "pages",
        "edit", "edits", "edited", "editing", "editor", "editors",
        "user", "users", "username", "block", "blocked", "blocking",
        "template", "unsigned", "utc", "http", "www", "com",
        "archive", "revision", "section", "category", "source",
        "deleted", "deletion", "admin", "administrator",
    })

    def __init__(self, slang: bool = False, verbose: bool = True):
        """
        Parameters
        ----------
        slang : bool
            Whether to apply slang normalization by default.
        verbose : bool
            Whether to print status messages when loading slang resources.
        """
        self.slang = slang
        self.verbose = verbose

        self.slang_map: dict[str, str] | None = None
        self.phrase_patterns: list[tuple[re.Pattern, str]] | None = None

        # Compiled regex patterns for standard cleaning
        self.re_url = re.compile(r"https?://\S+|www\.\S+")
        self.re_html = re.compile(r"<[^>]+>")
        self.re_newline = re.compile(r"[\r\n\t]+")
        self.re_repeated = re.compile(r"(.)\1{3,}")        # "fuuuuck" → "fuuck"
        self.re_punct = re.compile(r"[^\w\s']")            # strip punct, keep apostrophes
        self.re_stray_apos = re.compile(r"(?<!\w)'|'(?!\w)")
        self.re_spaces = re.compile(r"\s{2,}")

    def _first_sentence(self, text: str) -> str:
        """
        Return the first sentence of a description, stripped and lowercased.
        """
        text = text.strip()
        match = re.search(r"\.(?:\s|$)", text)

        if match:
            text = text[:match.start()]

        return text.lower().strip()

    def _load_slang_resources(self) -> tuple[dict[str, str], list[tuple[re.Pattern, str]]]:
        """
        Load MLBtrio/genz-slang-dataset and build:
        - slang_map: single-token slang dictionary
        - phrase_patterns: multi-word slang regex patterns

        Uses:
        - first-match-wins for duplicate slang terms
        - common-word filter to avoid corrupting normal English words
        """
        from datasets import load_dataset
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        dataset = load_dataset("MLBtrio/genz-slang-dataset", split="train")
        df = dataset.to_pandas()

        token_map: dict[str, str] = {}
        seen_phrases: set[str] = set()
        phrases: list[tuple[str, str]] = []

        for _, row in df.iterrows():
            slang = str(row["Slang"]).strip()
            description = self._first_sentence(str(row["Description"]))

            if not slang or not description:
                continue

            key = re.sub(r"[^a-z0-9\s]", "", slang.lower()).strip()

            if not key:
                continue

            # Avoid replacing common words like "an" or "was"
            if key in ENGLISH_STOP_WORDS:
                continue

            if " " in key:
                if key not in seen_phrases:
                    seen_phrases.add(key)
                    phrases.append((key, description))
            else:
                if key not in token_map:
                    token_map[key] = description

        # Longest phrase first so longer phrases win
        phrases.sort(key=lambda x: len(x[0]), reverse=True)

        phrase_patterns = [
            (re.compile(r"\b" + re.escape(phrase) + r"\b"), replacement)
            for phrase, replacement in phrases
        ]

        return token_map, phrase_patterns

    def _get_slang_resources(self) -> tuple[dict[str, str], list[tuple[re.Pattern, str]]]:
        """
        Return cached slang resources, loading them on first use.
        """
        if self.slang_map is None or self.phrase_patterns is None:
            if self.verbose:
                print("[TextPreprocessor] Loading slang dataset from HuggingFace...")

            self.slang_map, self.phrase_patterns = self._load_slang_resources()

            if self.verbose:
                print(
                    f"[TextPreprocessor] Loaded {len(self.slang_map)} single-token "
                    f"and {len(self.phrase_patterns)} multi-word slang entries."
                )

        return self.slang_map, self.phrase_patterns

    def _apply_slang_normalization(self, text: str) -> str:
        """
        Apply phrase-level then token-level slang normalization.

        Assumes text is already:
        - lowercased
        - punctuation-stripped
        - whitespace-normalized
        """
        token_map, phrase_patterns = self._get_slang_resources()

        # Multi-word phrase replacement first
        for pattern, replacement in phrase_patterns:
            text = pattern.sub(replacement, text)

        # Single-token replacement
        tokens = text.split()
        tokens = [token_map.get(token, token) for token in tokens]

        return " ".join(tokens)

    def clean_text(self, text: str, slang: bool | None = None) -> str:
        """
        Clean a single text string.

        Parameters
        ----------
        text : str
            Raw text.
        slang : bool | None
            Whether to apply slang normalization.
            If None, uses self.slang.

        Returns
        -------
        str
            Cleaned text.
        """
        if slang is None:
            slang = self.slang

        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = self.re_url.sub(" ", text)
        text = self.re_html.sub(" ", text)
        text = self.re_newline.sub(" ", text)
        text = self.re_repeated.sub(r"\1\1", text)
        text = self.re_punct.sub(" ", text)
        text = self.re_stray_apos.sub(" ", text)
        text = self.re_spaces.sub(" ", text).strip()

        if slang:
            text = self._apply_slang_normalization(text)
            text = self.re_spaces.sub(" ", text).strip()

        return text

    def preprocess_series(self, series: pd.Series, slang: bool | None = None) -> pd.Series:
        """
        Apply clean_text to a pandas Series.
        """
        return series.apply(lambda x: self.clean_text(x, slang=slang))

    def preprocess_df(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_col: str = "clean_text",
        slang: bool | None = None,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Clean a text column in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        text_col : str
            Name of the raw text column.
        output_col : str
            Name of the cleaned output column.
        slang : bool | None
            Whether to apply slang normalization.
        copy : bool
            Whether to return a copy or mutate the original DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned text column.
        """
        if copy:
            df = df.copy()

        df[output_col] = self.preprocess_series(df[text_col], slang=slang)
        return df
    
    def preprocess_series(
        self,
        series: pd.Series,
        slang: bool | None = None,
    ) -> pd.Series:
        """
        Apply clean_text to a pandas Series.
        """
        return series.apply(lambda x: self.clean_text(x, slang=slang)
    )

    def extract_length_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Extract character length and word count features.
        """
        return pd.DataFrame({
            "char_len": series.str.len().fillna(0).astype(float),
            "word_count": series.str.split().str.len().fillna(0).astype(float),
        })

    def slang_density(self, text: str) -> float:
        """
        Return the fraction of tokens that are known slang terms.
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0

        token_map, _ = self._get_slang_resources()

        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()

        if not tokens:
            return 0.0

        hits = sum(1 for token in tokens if token in token_map)
        return hits / len(tokens)

    def slang_density_series(self, series: pd.Series) -> pd.Series:
        """
        Apply slang_density to a pandas Series.
        """
        return series.apply(self.slang_density)