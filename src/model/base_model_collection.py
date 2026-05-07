from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseModelCollection(ABC):
    """
    Abstract base class for model collections.

    Any model collection must implement:
    - predict_individual
    - predict_confidence
    """

    @abstractmethod
    def predict_individual(self, X) -> dict[str, np.ndarray]:
        """
        Return hard predictions for each model.

        Returns
        -------
        dict[str, np.ndarray]
            model_name -> predicted labels
        """
        pass

    @abstractmethod
    def predict_confidence(self, X) -> dict[str, np.ndarray]:
        """
        Return toxic-class confidence scores for each model.

        For binary models, this should return probability/confidence for class 1.
        """
        pass