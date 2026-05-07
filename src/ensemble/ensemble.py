import joblib

import src.ensemble.simple_majority as sm
import src.ensemble.weighted_majority as wm
import src.ensemble.weighted_confidence_majority as wcm
import src.ensemble.stacked as st

from pathlib import Path
from sklearn.pipeline import Pipeline

class Ensemble:
    def __init__(self, classifiers_paths):
        if not  classifiers_paths:
            raise ValueError("At least one classifier or model path is required.")

        self.classifiers = self._load_classifiers(classifiers_paths)

        self.simple_majority = sm.SimpleMajority(self.classifiers)
        self.weighted_majority = wm.WeightedMajority(self.classifiers)

        self.weighted_confidence_majority = None
        self.stacked = None

    def _load_classifiers(self, classifiers_paths):
        """
        Load classifiers from joblib paths or keep already-loaded model objects.

        Supports:
        - raw sklearn classifiers
        - sklearn Pipelines
        - dictionaries containing a fitted pipeline under the key 'pipeline'
        """
        loaded_classifiers = []

        for clf in classifiers_paths:

            # Case 1: path to saved model
            if isinstance(clf, (str, Path)):
                clf_path = Path(clf)

                if clf_path.suffix != ".joblib":
                    raise ValueError(
                        f"Expected a .joblib file, got: {clf_path}"
                    )

                model = joblib.load(clf_path)

                # If saved object is a dictionary with a pipeline, extract pipeline
                if isinstance(model, dict) and "pipeline" in model:
                    model = model["pipeline"].named_steps["clf"]
                    
                loaded_classifiers.append(model)

            # Case 2: already-loaded model object
            else:
                model = clf
                loaded_classifiers.append(model)

            # verify models are fitted sklearn classifiers
            if not hasattr(model, "predict"):
                raise ValueError(
                    f"Loaded object does not appear to be a fitted sklearn model: {model}"
                )

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