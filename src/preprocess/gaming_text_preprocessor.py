from __future__ import annotations

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from tokenizer import tokenize


class GamingTextPreprocessor:
    """
    Handles TF-IDF preprocessing for social media / gaming text.

    Pipeline:
        raw text series -> tokenization -> TF-IDF sparse matrix
    """

    DEFAULT_TFIDF_KWARGS = {
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_df": 0.95,
        "sublinear_tf": True,
        "norm": "l2",
        "analyzer": "word",
        "tokenizer": tokenize,
        "token_pattern": None,
    }

    def __init__(self, tfidf_kwargs: dict | None = None):
        self.tfidf_kwargs = {
            **self.DEFAULT_TFIDF_KWARGS,
            **(tfidf_kwargs or {}),
        }

        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_kwargs)

    def fit(
        self,
        train_df: pd.DataFrame,
        text_column: str = "text",
    ) -> "GamingTextPreprocessor":
        """
        Fit the TF-IDF vectorizer on a training DataFrame.
        """
        self._validate_text_column(train_df, text_column)

        self.tfidf_vectorizer.fit(
            train_df[text_column].fillna("").astype(str)
        )

        return self

    def transform(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
    ) -> spmatrix:
        """
        Transform a DataFrame text column into TF-IDF features.
        """
        check_is_fitted(self.tfidf_vectorizer)
        self._validate_text_column(df, text_column)

        return self.tfidf_vectorizer.transform(
            df[text_column].fillna("").astype(str)
        )

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        text_column: str = "text",
    ) -> spmatrix:
        """
        Fit the TF-IDF vectorizer and transform the training data.
        """
        self._validate_text_column(train_df, text_column)

        return self.tfidf_vectorizer.fit_transform(
            train_df[text_column].fillna("").astype(str)
        )

    def process_text(self, text_series: pd.Series) -> spmatrix:
        """
        Transform a pandas Series of raw text into TF-IDF features.

        Kept as an alias for compatibility with your current code.
        """
        check_is_fitted(self.tfidf_vectorizer)

        return self.tfidf_vectorizer.transform(
            text_series.fillna("").astype(str)
        )

    @staticmethod
    def _validate_text_column(df: pd.DataFrame, text_column: str) -> None:
        """
        Validate that the requested text column exists.
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' was not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )