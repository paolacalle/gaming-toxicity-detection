from __future__ import annotations
import numpy as np

from src.model.bert_collection import BertToxicityModelCollection
from src.model.game_model_collection import GamingModelCollection
from src.model.social_media_collection import SocialMediaModelCollection
from src.ensemble.ensemble import Ensemble

class FixedWeightEnsembleModel(Ensemble):
    """
    An EnsembleModel with fixed weights that are set at initialization.

    This is a simple wrapper around EnsembleModel that allows for a more concise
    way to create an ensemble with fixed weights without needing to call
    set_weights() separately.
    """

    def __init__(
        self,
        model_collections: BertToxicityModelCollection | GamingModelCollection | SocialMediaModelCollection | list,
        weights: np.ndarray | list[float],
        threshold: float | None = None,
        name: str = "fixed_weight_ensemble_model",
    ):
        super().__init__(model_collections=model_collections)
        self.weights = weights
        self.threshold = threshold
        self.name = name
    
    def predict_simple_majority(self, X):
        return super().predict_simple_majority(X)
    
    def predict_weighted_majority(self, X):
        return super().predict_weighted_majority(X, weights=self.weights)
    
    def predict_weighted_confidence_majority(self, X):
        return super().predict_weighted_confidence_majority(
            X, 
            weights=self.weights,
            threshold=self.threshold
        )
        
    def predict_weighted_confidence_scores(self, X):
        return super().predict_weighted_confidence_scores(
            X, 
            weights=self.weights,
            # threshold=self.threshold
        )
        
    def predict(self, X):
        return self.predict_weighted_confidence_majority(X)