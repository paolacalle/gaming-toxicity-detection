import numpy as np


class WeightedMajority:
    def __init__(self, classifiers):
        if not classifiers:
            raise ValueError("At least one classifier is required.")

        self.classifiers = classifiers
        self.weights = np.ones(len(classifiers)) / len(classifiers)
        self.weight_history = []
        self.classes_ = None

    def predict(self, X, weights=None):
        """
        Weighted majority voting.

        Each classifier casts one vote for a class.
        The vote is weighted by that classifier's weight.
        The class with the largest total vote wins.
        """
        if weights is None:
            weights = self.weights

        weights = np.asarray(weights)

        if len(weights) != len(self.classifiers):
            raise ValueError("Number of weights must match number of classifiers.")

        preds = np.array([clf.predict(X) for clf in self.classifiers])

        # collect all possible predicted labels
        classes = np.unique(preds)

        weighted_votes = np.zeros((X.shape[0], len(classes)))

        for model_idx, weight in enumerate(weights):
            for class_idx, cls in enumerate(classes):
                weighted_votes[:, class_idx] += weight * (preds[model_idx] == cls)

        return classes[weighted_votes.argmax(axis=1)]

    def fit_weights_random_search(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42
    ):
        """
        Finds ensemble weights by randomly sampling weight combinations
        and keeping the weights with the best validation score.
        """
        rng = np.random.default_rng(random_state)

        best_score = -np.inf
        best_weights = self.weights.copy()

        self.weight_history = []

        for _ in range(n_trials):
            weights = rng.dirichlet(np.ones(len(self.classifiers)))

            predictions = self.predict(X_val, weights)

            score = score_func(y_val, predictions)

            self.weight_history.append((weights.copy(), score))

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        self.weights = best_weights

        return best_weights, best_score