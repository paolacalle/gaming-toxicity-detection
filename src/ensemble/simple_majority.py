import numpy as np


class SimpleMajority:
    def __init__(self, models):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models

    def predict(self, X):
        """
        Simple majority voting.

        Each model casts one vote.
        The class with the most votes is selected.
        """
        print("Predicting with SimpleMajority...")
        predictions = np.array([model.predict(X) for model in self.models])

        majority_votes = []

        for i in range(len(X)):
            labels, counts = np.unique(predictions[:, i], return_counts=True)
            majority_label = labels[np.argmax(counts)]
            majority_votes.append(majority_label)

        return np.array(majority_votes)