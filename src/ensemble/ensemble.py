import simple_majority as sm
import weighted_majority as wm
import weighted_confidence_majority as wcm
import stacked as st


class Ensemble:
    def __init__(self, classifiers):
        if not classifiers:
            raise ValueError("At least one classifier is required.")

        self.classifiers = classifiers

        self.simple_majority = sm.SimpleMajority(classifiers)
        self.weighted_majority = wm.WeightedMajority(classifiers)

        self.weighted_confidence_majority = None
        self.stacked = None

    def predict_simple_majority(self, X):
        """
        Predict using simple majority voting.
        """
        return self.simple_majority.predict(X)

    def predict_weighted_majority(self, X, weights=None):
        """
        Predict using weighted majority voting.
        """
        return self.weighted_majority.predict(X, weights)

    def fit_weighted_majority(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42
    ):
        """
        Tune weighted majority voting weights using random search.
        """
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
        """
        Initialize and tune weighted confidence averaging using random search.
        """
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
        """
        Predict using weighted confidence averaging.
        """
        if self.weighted_confidence_majority is None:
            self.weighted_confidence_majority = wcm.WeightedConfidenceMajority(
                self.classifiers
            )

        return self.weighted_confidence_majority.predict_weighted(X, weights)

    def fit_stacked(self, X, y, meta_model, n_folds=5, random_state=42):
        """
        Fit stacked ensemble using self.classifiers as base models.
        """
        self.stacked = st.Stacked(
            base_models=self.classifiers,
            meta_model=meta_model,
            n_folds=n_folds,
            random_state=random_state
        )

        self.stacked.fit(X, y)

        return self

    def predict_stacked(self, X):
        """
        Predict using stacked ensemble.
        """
        if self.stacked is None:
            raise ValueError("Stacked model has not been fitted yet.")

        return self.stacked.predict(X)