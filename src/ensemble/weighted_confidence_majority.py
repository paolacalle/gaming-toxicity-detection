import numpy as np


class WeightedConfidenceMajority:
    def __init__(self, model_collections):
        if not model_collections:
            raise ValueError("At least one model collection is required.")

        self.model_collections = model_collections
        self.model_names = None
        self.weights = None
        self.weight_history = []
        self.threshold = None
        self.classes_ = None

    def _get_confidences_dict(self, X):
        """
        Collect confidence outputs from all model collections.

        Each collection must return:
            model_name -> confidence array

        Binary allowed shapes:
            (n_samples,)      toxic score only
            (n_samples, 2)    class probabilities

        Multiclass required shape:
            (n_samples, n_classes)
        """
        print("Collecting confidence outputs from models...")
        confidences_dict = {}

        for collection in self.model_collections:
            collection_confidences = collection.predict_confidence(X)

            for model_name, scores in collection_confidences.items():
                if model_name in confidences_dict:
                    raise ValueError(f"Duplicate model name found: {model_name}")

                scores = np.asarray(scores, dtype=float)

                # Binary toxic-score vector -> convert to 2-column probabilities
                if scores.ndim == 1:
                    scores = np.column_stack([1 - scores, scores])

                if scores.ndim != 2:
                    raise ValueError(
                        f"{model_name} confidence output must be 1D or 2D, "
                        f"got shape {scores.shape}."
                    )

                confidences_dict[model_name] = scores
                
        print("Total models in ensemble:", len(confidences_dict))

        return confidences_dict

    def _get_confidence_tensor(self, X):
        """
        Returns
        -------
        model_names : list[str]

        confidence_tensor : np.ndarray
            Shape: (n_models, n_samples, n_classes)
        """
        print("Constructing confidence tensor...")
        confidences_dict = self._get_confidences_dict(X)

        model_names = list(confidences_dict.keys())
        matrices = [confidences_dict[name] for name in model_names]

        n_samples = matrices[0].shape[0]
        n_classes = matrices[0].shape[1]
        
        print(f"Expected confidence shape: (n_samples={n_samples}, n_classes={n_classes})")

        for name, matrix in zip(model_names, matrices):
            if matrix.shape[0] != n_samples:
                raise ValueError(
                    f"All models must predict the same number of samples. "
                    f"Expected {n_samples}, but {name} has {matrix.shape[0]}."
                )

            if matrix.shape[1] != n_classes:
                raise ValueError(
                    f"All models must output the same number of classes. "
                    f"Expected {n_classes}, but {name} has {matrix.shape[1]}."
                )

        self.model_names = model_names
        self.classes_ = np.arange(n_classes)

        return model_names, np.stack(matrices, axis=0)

    def _ensure_weights(self, model_names, weights=None):
        """
        Ensures weights are aligned with model_names.

        weights can be:
            - None
            - list / np.ndarray in the exact model_names order
            - dict mapping model_name -> weight
        """
        if weights is None:
            if self.weights is None:
                weights = np.ones(len(model_names)) / len(model_names)
            else:
                weights = self.weights

        # Case 1: weights passed as dictionary
        if isinstance(weights, dict):
            print("Ensuring weights from dictionary...")
            missing = set(model_names) - set(weights.keys())
            extra = set(weights.keys()) - set(model_names)

            if missing:
                raise ValueError(
                    f"Missing weights for models: {sorted(missing)}. "
                    f"Expected model names: {model_names}"
                )

            if extra:
                raise ValueError(
                    f"Got weights for unknown models: {sorted(extra)}. "
                    f"Expected model names: {model_names}"
                )
                
            print("Model names and weight keys match. Constructing weight array...")
            print("Respecting order of model names...")

            weights = np.array([weights[name] for name in model_names], dtype=float)

        # Case 2: weights passed as list / ndarray
        else:
            weights = np.asarray(weights, dtype=float)

            if len(weights) != len(model_names):
                raise ValueError(
                    f"Number of weights must match number of models. "
                    f"Got {len(weights)} weights for {len(model_names)} models. "
                    f"Model order is: {model_names}"
                )

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
        """
        model_names, confidence_tensor = self._get_confidence_tensor(X)

        avg_probas = confidence_tensor.mean(axis=0)
        return self.classes_[avg_probas.argmax(axis=1)]

    def predict_weighted(self, X, weights=None, threshold=None, uncertain_label=-1):
        """
        Weighted confidence averaging.

        Multiclass:
            weighted probabilities -> argmax class
            if threshold is provided, only accept prediction when max confidence >= threshold

        Binary:
            same logic by default, unless you specifically want class-1 thresholding.
        """
        model_names, confidence_tensor = self._get_confidence_tensor(X)
        weights = self._ensure_weights(model_names, weights)

        weighted_probas = np.tensordot(
            weights,
            confidence_tensor,
            axes=(0, 0)
        )
        # shape: (n_samples, n_classes)
        predicted_labels = self.classes_[weighted_probas.argmax(axis=1)]
        
        print(f"Predicted labels shape: {predicted_labels.shape}")

        if threshold is not None:
            max_confidence = weighted_probas.max(axis=1)
            predicted_labels = np.where(
                max_confidence >= threshold,
                predicted_labels,
                uncertain_label
            )

        return predicted_labels

    def fit_weights_random_search(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42,
        thresholds=None,
        uncertain_label=-1,
    ):
        """
        Random-search weights for confidence averaging.

        For both binary and multiclass:
            1. Compute weighted class probabilities.
            2. Predict argmax class.
            3. If threshold is provided, only accept the predicted label when
            max class confidence >= threshold.
            4. Otherwise return uncertain_label.

        score_func should handle uncertain_label if thresholds are used.
        """
        rng = np.random.default_rng(random_state)

        print("Collecting confidence outputs from models...")
        model_names, confidence_tensor = self._get_confidence_tensor(X_val)

        n_models = len(model_names)
        
        print("Starting random search for n models:", n_models)
        y_val = np.asarray(y_val)

        best_score = -np.inf
        best_weights = np.ones(n_models) / n_models
        best_threshold = None

        self.weight_history = []

        if thresholds is None:
            thresholds = [None]

        for _ in range(n_trials):
            # sample from simplex to get nonnegative weights that sum to 1
            weights = rng.dirichlet(np.ones(n_models))

            weighted_probas = np.tensordot(
                weights,
                confidence_tensor,
                axes=(0, 0)
            )
            
            print(f"Trial weights: {weights.shape}")
            print(f"Weighted probabilities shape: {weighted_probas.shape}")
            
            # shape: (n_samples, n_classes)

            predicted_labels = self.classes_[weighted_probas.argmax(axis=1)]
            max_confidence = weighted_probas.max(axis=1)

            for threshold in thresholds:
                if threshold is not None:
                    predictions = np.where(
                        max_confidence >= threshold,
                        predicted_labels,
                        uncertain_label
                    )
                else:
                    predictions = predicted_labels

                score = score_func(y_val, predictions)
                coverage = (predictions != uncertain_label).mean()
                accuracy = (predictions == y_val).mean()

                self.weight_history.append({
                    "weights": weights.copy(),
                    "threshold": threshold,
                    "score": score,
                    "coverage": coverage,
                    "acc": accuracy,
                })

                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    best_threshold = threshold
                    
        # best weights dict so you know 
        # which model got which weight in the best combination
        best_weights_dict = dict(zip(model_names, best_weights))
        print("Best weights:", best_weights_dict)

        self.weights = best_weights
        self.threshold = best_threshold

        return best_weights, best_threshold, best_score, model_names