import numpy as np


class WeightedConfidenceMajority:
    def __init__(self, classifiers):
        if not classifiers:
            raise ValueError("At least one classifier is required.")

        self.classifiers = classifiers
        self.weights = np.ones(len(classifiers)) / len(classifiers)
        self.weight_history = []

        # assume all classifiers were trained on the same classes
        if not hasattr(classifiers[0], "classes_"):
            raise ValueError("Classifiers must be fitted and have a classes_ attribute.")

        self.classes_ = classifiers[0].classes_

    def _get_probas(self, X):
        """
        Gets predicted probabilities from each classifier.

        Assumes all classifiers return probabilities in the same class order.
        """
        probas = []

        for clf in self.classifiers:
            if not hasattr(clf, "predict_proba"):
                raise ValueError(f"{clf.__class__.__name__} does not support predict_proba.")

            if not np.array_equal(clf.classes_, self.classes_):
                raise ValueError("All classifiers must have the same class order.")

            probas.append(clf.predict_proba(X))

        return probas

    def predict(self, X):
        """
        Unweighted confidence averaging.

        Each classifier contributes equally.
        """
        probas = self._get_probas(X)
        avg_probas = np.mean(probas, axis=0)

        return self.classes_[avg_probas.argmax(axis=1)]

    def predict_weighted(self, X, weights=None):
        """
        Weighted confidence averaging.

        Each classifier's probability distribution is weighted before averaging.
        """
        if weights is None:
            weights = self.weights

        weights = np.asarray(weights)

        if len(weights) != len(self.classifiers):
            raise ValueError("Number of weights must match number of classifiers.")

        if np.any(weights < 0):
            raise ValueError("Weights must be nonnegative.")

        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

        probas = self._get_probas(X)

        weighted_probas = sum(w * p for w, p in zip(weights, probas))

        return self.classes_[weighted_probas.argmax(axis=1)]

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

        score_func should take:
            score_func(y_true, y_pred)

        Example:
            sklearn.metrics.f1_score with average="macro"
        """
        rng = np.random.default_rng(random_state)

        probas = self._get_probas(X_val)

        best_score = -np.inf
        best_weights = self.weights.copy()

        self.weight_history = []

        for _ in range(n_trials):
            weights = rng.dirichlet(np.ones(len(self.classifiers)))

            weighted_probas = sum(w * p for w, p in zip(weights, probas))
            predictions = self.classes_[weighted_probas.argmax(axis=1)]

            score = score_func(y_val, predictions)

            self.weight_history.append((weights.copy(), score))

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        self.weights = best_weights

        return best_weights, best_score