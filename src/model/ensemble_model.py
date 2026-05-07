from __future__ import annotations

import numpy as np

from src.model.base_model_collection import BaseModelCollection
from src.model.bert_collection import BertToxicityModelCollection
from src.model.game_model_collection import GamingModelCollection
from src.model.social_media_collection import SocialMediaModelCollection
from src.ensemble.ensemble import Ensemble


ModelCollection = (
    BertToxicityModelCollection
    | GamingModelCollection
    | SocialMediaModelCollection
)


class EnsembleModel(BaseModelCollection):
    """
    Wraps an Ensemble so it can behave like a BaseModelCollection.

    This lets an ensemble itself be used as a model collection inside
    another ensemble if needed.
    """

    def __init__(
        self,
        model_collections: list[ModelCollection],
        weights: np.ndarray | list[float] | None = None,
        threshold: float | None = None,
        name: str = "ensemble_model",
    ):
        if not model_collections:
            raise ValueError("At least one model collection is required.")

        self.model_collections = model_collections
        self.ensemble = Ensemble(model_collections)

        self.weights = weights
        self.threshold = threshold
        self.name = name

    def set_weights(self, weights):
        self.weights = weights
        return self

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        return self

    def predict(self, X) -> np.ndarray:
        """
        Default prediction method.

        Uses weighted confidence if weights are set.
        Otherwise falls back to simple majority.
        """
        if self.weights is not None:
            return self.predict_weighted_confidence_majority(X)

        return self.ensemble.predict_simple_majority(X)

    def predict_individual(self, X) -> dict[str, np.ndarray]:
        """
        Required by BaseModelCollection.

        Since this wrapper represents one ensemble model, return:
            ensemble_name -> predictions
        """
        return {
            self.name: self.predict(X)
        }

    def predict_simple_majority(self, X) -> np.ndarray:
        return self.ensemble.predict_simple_majority(X)

    def predict_weighted_majority(self, X) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights must be set to use weighted majority prediction.")

        return self.ensemble.predict_weighted_majority(
            X,
            weights=self.weights,
        )

    def predict_weighted_confidence_majority(self, X) -> np.ndarray:
        if self.weights is None:
            raise ValueError(
                "Weights must be set to use weighted confidence majority prediction."
            )

        return self.ensemble.predict_weighted_confidence_majority(
            X,
            weights=self.weights,
            threshold=self.threshold,
        )

    def predict_weighted_confidence_scores(self, X) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights must be set to use weighted confidence scores.")

        return self.ensemble.predict_weighted_confidence_scores(
            X,
            weights=self.weights,
        )

    def predict_confidence(self, X) -> dict[str, np.ndarray]:
        """
        Required by BaseModelCollection.

        Returns ensemble confidence scores as:
            ensemble_name -> scores
        """
        return {
            self.name: self.predict_weighted_confidence_scores(X)
        }