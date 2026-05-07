from __future__ import annotations

from typing import Optional
import pandas as pd


class GamingTextLabeler:
    DEFAULT_WOT_SCHEME: dict[int, int] = {
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
    }

    DEFAULT_DOTA_SCHEME: dict[int, int] = {
        0: 0,
        1: 2,
        2: 2,
        3: 1,
    }

    def __init__(
        self,
        wot_scheme: Optional[dict[int, int]] = None,
        dota_scheme: Optional[dict[int, int]] = None,
    ):
        self.wot_scheme = wot_scheme or self.DEFAULT_WOT_SCHEME
        self.dota_scheme = dota_scheme or self.DEFAULT_DOTA_SCHEME
        self.class_names = ["Non-Toxic", "Mild Toxicity", "Severe Toxicity"]

    def convert_binary(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
        output_column: str | None = None,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Convert labels to binary toxicity labels.

        0 -> 0 non-toxic
        nonzero -> 1 toxic
        """
        if copy:
            df = df.copy()

        output_column = output_column or label_column

        df[output_column] = df[label_column].apply(
            lambda x: 0 if x == 0 else 1
        )

        return df

    def convert_three_class(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
        output_column: str | None = None,
        data_source_column: str = "data_source",
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Convert labels to three-class toxicity labels.

        WoT:
            0 -> 0 non-toxic
            1, 2 -> 1 mild toxicity
            3, 4, 5 -> 2 severe toxicity

        DOTA:
            0 -> 0 non-toxic
            1, 2 -> 2 severe toxicity
            3 -> 1 mild toxicity
        """
        if copy:
            df = df.copy()

        output_column = output_column or label_column

        # base on data source, apply the appropriate mapping scheme
        df[output_column] = df.apply(
            lambda row: self._get_scheme(row[data_source_column]).get(
                row[label_column], row[label_column]
            ),
            axis=1,
        ).astype(int)

        return df

    def _get_scheme(self, data_source: str) -> dict[int, int]:
        """
        Return the label mapping scheme for the selected data source.
        """
        if data_source == "wot":
            return self.wot_scheme

        if data_source == "dota":
            return self.dota_scheme

        raise ValueError(
            f"Unknown data source: {data_source}. Expected 'wot' or 'dota'."
        )