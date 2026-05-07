import numpy as np


class SimpleMajority:
    def __init__(self, model_collections):
        if not model_collections:
            raise ValueError("At least one model collection is required.")

        self.model_collections = model_collections

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

                predictions_dict[model_name] = np.asarray(preds)

        return predictions_dict

    def predict(self, X):
        """
        Simple majority voting.

        Each model inside each collection casts one vote.
        The class with the most votes is selected.
        """
        print("Predicting with SimpleMajority...")

        predictions_dict = self._get_predictions_dict(X)

        predictions = np.vstack(list(predictions_dict.values()))
        # shape: (n_models, n_samples)

        majority_votes = []

        for i in range(predictions.shape[1]):
            labels, counts = np.unique(predictions[:, i], return_counts=True)
            majority_label = labels[np.argmax(counts)]
            majority_votes.append(majority_label)

        return np.array(majority_votes)