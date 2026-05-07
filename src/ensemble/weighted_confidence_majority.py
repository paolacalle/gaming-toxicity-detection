import numpy as np


class WeightedConfidenceMajority:
    def __init__(self, model_collections):
        if not model_collections:
            raise ValueError("At least one model collection is required.")

        self.model_collections = model_collections
        self.model_names = None
        self.weights = None
        self.weight_history = []
        self.threshold = 0.5

        # Binary setting by design because predict_confidence returns toxic score
        self.classes_ = np.array([0, 1])

    def _get_confidences_dict(self, X):
        """
        Collect toxic-class confidence scores from all model collections.

        Each collection must implement:
            predict_confidence(X) -> dict[str, np.ndarray]
        """
        confidences_dict = {}

        for collection in self.model_collections:
            collection_confidences = collection.predict_confidence(X)

            for model_name, scores in collection_confidences.items():
                if model_name in confidences_dict:
                    raise ValueError(f"Duplicate model name found: {model_name}")

                confidences_dict[model_name] = np.asarray(scores, dtype=float)

        return confidences_dict

    def _get_confidence_matrix(self, X):
        """
        Returns
        -------
        model_names : list[str]
        confidence_matrix : np.ndarray
            Shape: (n_samples, n_models)
        """
        confidences_dict = self._get_confidences_dict(X)

        model_names = list(confidences_dict.keys())
        confidence_matrix = np.vstack(
            [confidences_dict[name] for name in model_names]
        ).T

        return model_names, confidence_matrix

    def _ensure_weights(self, model_names, weights=None):
        """
        Validate/initialize weights.
        """
        if weights is None:
            if self.weights is None or len(self.weights) != len(model_names):
                self.weights = np.ones(len(model_names)) / len(model_names)

            weights = self.weights

        weights = np.asarray(weights, dtype=float)

        if len(weights) != len(model_names):
            raise ValueError("Number of weights must match number of models.")

        if np.any(weights < 0):
            raise ValueError("Weights must be nonnegative.")

        if weights.sum() == 0:
            raise ValueError("At least one weight must be nonzero.")

        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

        return weights

    def predict(self, X):
        """
        Unweighted confidence averaging.

        Each model contributes equally.
        """
        model_names, confidence_matrix = self._get_confidence_matrix(X)

        avg_scores = confidence_matrix.mean(axis=1)

        return (avg_scores >= self.threshold).astype(int)

    def predict_weighted(self, X, weights=None, threshold=None):
        """
        Weighted confidence averaging.

        Each model's toxic confidence score is weighted before averaging.
        """
        if threshold is None:
            threshold = self.threshold

        model_names, confidence_matrix = self._get_confidence_matrix(X)
        weights = self._ensure_weights(model_names, weights)

        weighted_scores = confidence_matrix @ weights

        return (weighted_scores >= threshold).astype(int)

    def predict_scores(self, X, weights=None):
        """
        Return weighted confidence scores without thresholding.
        Useful for threshold tuning.
        """
        model_names, confidence_matrix = self._get_confidence_matrix(X)
        weights = self._ensure_weights(model_names, weights)

        return confidence_matrix @ weights

    def fit_weights_random_search(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42,
        thresholds=None,
    ):
        """
        Finds ensemble weights by randomly sampling weight combinations
        and keeping the weights with the best validation score.

        Also optionally tunes threshold.

        score_func should take:
            score_func(y_true, y_pred)
        """
        rng = np.random.default_rng(random_state)

        model_names, confidence_matrix = self._get_confidence_matrix(X_val)
        n_models = len(model_names)

        if thresholds is None:
            thresholds = [self.threshold]

        best_score = -np.inf
        best_weights = np.ones(n_models) / n_models
        best_threshold = self.threshold

        self.weight_history = []

        for _ in range(n_trials):
            weights = rng.dirichlet(np.ones(n_models))
            weighted_scores = confidence_matrix @ weights

            for threshold in thresholds:
                predictions = (weighted_scores >= threshold).astype(int)

                score = score_func(y_val, predictions)

                self.weight_history.append(
                    {
                        "weights": weights.copy(),
                        "threshold": threshold,
                        "score": score,
                    }
                )

                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    best_threshold = threshold

        self.weights = best_weights
        self.threshold = best_threshold

        return best_weights, best_score