from __future__ import annotations

from click import Tuple
from nltk.featstruct import TYPE
import numpy as np

from src.model.base_model_collection import BaseModelCollection
from src.model.fixed_model_ensemble import FixedWeightEnsembleModel


class EnsembleModel(BaseModelCollection):
    """
    Combines multiple FixedWeightEnsembleModel objects into a higher-level ensemble.

    Important:
    This class combines confidence/probability scores, not class labels.
    """

    def __init__(
        self,
        model_ensembles: list[FixedWeightEnsembleModel],
        weights: np.ndarray | list[float] | None = None,
        name: str = "ensemble_model",
    ):
        if not model_ensembles:
            raise ValueError("At least one model ensemble is required.")

        self.model_ensembles = model_ensembles
        self.name = name

        if weights is not None:
            weights = np.asarray(weights, dtype=float)

            if len(weights) != len(model_ensembles):
                raise ValueError(
                    f"Expected {len(model_ensembles)} weights, got {len(weights)}."
                )

            if weights.sum() <= 0:
                raise ValueError("Weights must sum to a positive value.")

            weights = weights / weights.sum()

        self.weights = weights

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=float)

        if len(weights) != len(self.model_ensembles):
            raise ValueError(
                f"Expected {len(self.model_ensembles)} weights, got {len(weights)}."
            )

        if weights.sum() <= 0:
            raise ValueError("Weights must sum to a positive value.")

        self.weights = weights / weights.sum()
        return self

    def predict_weighted_confidence_scores(self, X) -> np.ndarray:
        """
        Average confidence/probability scores from each child ensemble.

        Expected child output:
            shape (n_samples, n_classes)
        """
        scores = []

        for ensemble in self.model_ensembles:
            ensemble_scores = ensemble.predict_weighted_confidence_scores(X)

            if ensemble_scores.ndim != 2:
                raise ValueError(
                    f"{ensemble.name} must return a 2D confidence matrix, "
                    f"got shape {ensemble_scores.shape}."
                )

            scores.append(ensemble_scores)

        scores = np.asarray(scores)
        # shape: (n_ensembles, n_samples, n_classes)

        if self.weights is not None:
            final_scores = np.average(scores, axis=0, weights=self.weights)
        else:
            final_scores = np.mean(scores, axis=0)

        return final_scores

    def predict(self, X) -> np.ndarray:
        """
        Final class prediction from majority or weighted-majority votes
        over child ensemble predictions.

        Works for binary and multiclass.
        """
        votes = np.asarray([
            ensemble.predict(X) for ensemble in self.model_ensembles
        ])

        # shape: (n_ensembles, n_samples)
        n_samples = votes.shape[1]
        final_votes = np.empty(n_samples, dtype=votes.dtype)

        for i in range(n_samples):
            ensemble_votes = votes[:, i]
            unique_classes = np.unique(ensemble_votes)

            if self.weights is not None:
                class_scores = np.array([
                    self.weights[ensemble_votes == cls].sum()
                    for cls in unique_classes
                ])
            else:
                class_scores = np.array([
                    np.sum(ensemble_votes == cls)
                    for cls in unique_classes
                ])

            final_votes[i] = unique_classes[np.argmax(class_scores)]

        return final_votes

    def predict_individual(self, X) -> dict[str, np.ndarray]:
        return {
            self.name: self.predict(X)
        }

    def predict_confidence(self, X) -> dict[str, np.ndarray]:
        return {
            self.name: self.predict_weighted_confidence_scores(X)
        }