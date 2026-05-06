import joblib

import simple_majority as sm
import weighted_majority as wm
import weighted_confidence_majority as wcm
import stacked as st


class Ensemble:
    def __init__(self, classifiers):
        if not classifiers:
            raise ValueError("At least one classifier or model path is required.")

        self.classifiers = self._load_classifiers(classifiers)

        self.simple_majority = sm.SimpleMajority(self.classifiers)
        self.weighted_majority = wm.WeightedMajority(self.classifiers)

        self.weighted_confidence_majority = None
        self.stacked = None

    def _load_classifiers(self, classifiers):
        """
        Loads classifiers if file paths are provided.
        Otherwise, keeps already-loaded model objects.
        """
        loaded_classifiers = []

        for clf in classifiers:
            if isinstance(clf, str) and clf.endswith(".joblib"):
                loaded_classifiers.append(joblib.load(clf))
            else:
                loaded_classifiers.append(clf)

        return loaded_classifiers

    def predict_simple_majority(self, X):
        return self.simple_majority.predict(X)

    def predict_weighted_majority(self, X, weights=None):
        return self.weighted_majority.predict(X, weights)

    def fit_weighted_majority(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42
    ):
        return self.weighted_majority.fit_weights_random_search(
            X_val=X_val,
            y_val=y_val,
            score_func=score_func,
            n_trials=n_trials,
            random_state=random_state
        )

    def fit_weighted_confidence_majority(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42
    ):
        self.weighted_confidence_majority = wcm.WeightedConfidenceMajority(
            self.classifiers
        )

        return self.weighted_confidence_majority.fit_weights_random_search(
            X_val=X_val,
            y_val=y_val,
            score_func=score_func,
            n_trials=n_trials,
            random_state=random_state
        )

    def predict_weighted_confidence_majority(self, X, weights=None):
        if self.weighted_confidence_majority is None:
            self.weighted_confidence_majority = wcm.WeightedConfidenceMajority(
                self.classifiers
            )

        return self.weighted_confidence_majority.predict_weighted(X, weights)

    def fit_stacked(self, X, y, meta_model, n_folds=5, random_state=42):
        self.stacked = st.Stacked(
            base_models=self.classifiers,
            meta_model=meta_model,
            n_folds=n_folds,
            random_state=random_state
        )

        self.stacked.fit(X, y)

        return self

    def predict_stacked(self, X):
        if self.stacked is None:
            raise ValueError("Stacked model has not been fitted yet.")

        return self.stacked.predict(X)