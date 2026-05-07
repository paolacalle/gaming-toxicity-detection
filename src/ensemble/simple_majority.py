import numpy as np


class SimpleMajority:
    def __init__(self, model_collections):
        if not model_collections:
            raise ValueError("At least one model collection is required.")

        self.model_collections = model_collections

    def _to_single_label(self, preds):
        """
        Convert predictions to 1D class labels.

        Handles:
        - already single-label predictions: shape (n_samples,)
        - multi-output binary predictions: shape (n_samples, n_outputs)

        For multi-output:
            [0, 0] -> 0
            [1, 0] -> 1
            [0, 1] -> 2
            [1, 1] -> highest active label + 1
        """
        preds = np.asarray(preds)

        if preds.ndim == 1:
            return preds

        if preds.ndim != 2:
            raise ValueError(
                f"Predictions must be 1D or 2D. Got shape {preds.shape}."
            )

        single_labels = np.zeros(preds.shape[0], dtype=int)

        for i, row in enumerate(preds):
            active = np.where(row == 1)[0]

            if len(active) == 0:
                single_labels[i] = 0
            else:
                single_labels[i] = active.max() + 1

        return single_labels

    def _get_predictions_dict(self, X):
        """
        Collect predictions from all model collections.

        Each collection must implement:
            predict_individual(X) -> dict[str, np.ndarray]
        """
        predictions_dict = {}

        for collection in self.model_collections:
            collection_preds = collection.predict_individual(X)

            for model_name, preds in collection_preds.items():
                if model_name in predictions_dict:
                    raise ValueError(f"Duplicate model name found: {model_name}")

                preds = self._to_single_label(preds)

                predictions_dict[model_name] = preds

                print(f"{model_name}: {preds.shape}")

        print(f"Collected predictions from {len(predictions_dict)} models.")

        return predictions_dict

    def predict(self, X):
        """
        Simple majority voting.

        Each model inside each collection casts one vote per sample.
        The class with the most votes is selected.
        """
        print("Predicting with SimpleMajority...")

        pred_len = len(X)
        print(f"Input has {pred_len} samples.")

        predictions_dict = self._get_predictions_dict(X)

        # shape: (n_samples, n_models)
        predictions = np.column_stack(list(predictions_dict.values()))

        print(f"Prediction matrix shape: {predictions.shape}")
        print(f"Aggregating predictions from {predictions.shape[1]} models...")

        majority_votes = []

        for i in range(predictions.shape[0]):
            labels, counts = np.unique(predictions[i, :], return_counts=True)
            majority_label = labels[np.argmax(counts)]
            majority_votes.append(majority_label)

        return np.array(majority_votes)