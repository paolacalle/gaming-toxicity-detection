import numpy as np


class WeightedMajority:
    def __init__(self, model_collections):
        if not model_collections:
            raise ValueError("At least one model collection is required.")

        self.model_collections = model_collections
        self.model_names = None
        self.weights = None
        self.weight_history = []

    def _get_predictions_dict(self, X):
        """
        Collect predictions from all model collections.
        """
        predictions_dict = {}

        for collection in self.model_collections:
            collection_preds = collection.predict_individual(X)

            for model_name, preds in collection_preds.items():
                if model_name in predictions_dict:
                    raise ValueError(f"Duplicate model name found: {model_name}")

                predictions_dict[model_name] = np.asarray(preds)

        return predictions_dict

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

    def predict(self, X, weights=None):
        """
        Weighted majority voting.

        Each model casts one hard-label vote.
        The vote is weighted by that model's weight.
        The class with the largest weighted vote wins.
        """
        predictions_dict = self._get_predictions_dict(X)

        model_names = list(predictions_dict.keys())
        self.model_names = model_names

        weights = self._ensure_weights(model_names, weights)

        predictions = np.vstack([predictions_dict[name] for name in model_names])
        # shape: (n_models, n_samples)

        classes = np.unique(predictions)
        n_samples = predictions.shape[1]

        weighted_votes = np.zeros((n_samples, len(classes)))

        for model_idx, weight in enumerate(weights):
            for class_idx, cls in enumerate(classes):
                weighted_votes[:, class_idx] += weight * (
                    predictions[model_idx] == cls
                )

        return classes[weighted_votes.argmax(axis=1)]

    def fit_weights_random_search(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42,
    ):
        """
        Finds ensemble weights by randomly sampling weight combinations
        and keeping the weights with the best validation score.
        """
        rng = np.random.default_rng(random_state)

        predictions_dict = self._get_predictions_dict(X_val)
        model_names = list(predictions_dict.keys())
        self.model_names = model_names

        predictions_matrix = np.vstack(
            [predictions_dict[name] for name in model_names]
        )
        classes = np.unique(predictions_matrix)
        n_models = len(model_names)
        n_samples = predictions_matrix.shape[1]

        best_score = -np.inf
        best_weights = np.ones(n_models) / n_models

        self.weight_history = []

        for _ in range(n_trials):
            weights = rng.dirichlet(np.ones(n_models))

            weighted_votes = np.zeros((n_samples, len(classes)))

            for model_idx, weight in enumerate(weights):
                for class_idx, cls in enumerate(classes):
                    weighted_votes[:, class_idx] += weight * (
                        predictions_matrix[model_idx] == cls
                    )

            predictions = classes[weighted_votes.argmax(axis=1)]

            score = score_func(y_val, predictions)

            self.weight_history.append((weights.copy(), score))

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        self.weights = best_weights

        return best_weights, best_score