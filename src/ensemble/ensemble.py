import src.ensemble.simple_majority as sm
import src.ensemble.weighted_majority as wm
import src.ensemble.weighted_confidence_majority as wcm
import src.ensemble.stacked as st

class Ensemble:
    def __init__(self, model_collections: list):
        if not model_collections:
            raise ValueError("At least one model collection is required.")

        self.model_collections = model_collections

        for collection in self.model_collections:
            if not hasattr(collection, "predict_individual"):
                raise ValueError(
                    f"{collection} does not implement predict_individual."
                )

            if not hasattr(collection, "predict_confidence"):
                raise ValueError(
                    f"{collection} does not implement predict_confidence."
                )

        self.simple_majority = sm.SimpleMajority(self.model_collections)
        self.weighted_majority = wm.WeightedMajority(self.model_collections)

        self.weighted_confidence_majority = None
        self.stacked = None

    def predict_simple_majority(self, X):
        return self.simple_majority.predict(X)

    def predict_weighted_majority(self, X, weights=None):
        return self.weighted_majority.predict(X, weights)
    
    # ---- Weight fitting and prediction for Weighted Confidence Majority ----

    def fit_weighted_majority(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42,
    ):
        return self.weighted_majority.fit_weights_random_search(
            X_val=X_val,
            y_val=y_val,
            score_func=score_func,
            n_trials=n_trials,
            random_state=random_state,
        )

    def fit_weighted_confidence_majority(
        self,
        X_val,
        y_val,
        score_func,
        n_trials=1000,
        random_state=42,
        thresholds=None,
    ):
        self.weighted_confidence_majority = wcm.WeightedConfidenceMajority(
            self.model_collections
        )

        return self.weighted_confidence_majority.fit_weights_random_search(
            X_val=X_val,
            y_val=y_val,
            score_func=score_func,
            n_trials=n_trials,
            random_state=random_state,
            thresholds=thresholds,
        )

    def predict_weighted_confidence_majority(
        self,
        X,
        weights=None,
        threshold=None,
    ):
        if self.weighted_confidence_majority is None:
            self.weighted_confidence_majority = wcm.WeightedConfidenceMajority(
                self.model_collections
            )

        return self.weighted_confidence_majority.predict_weighted(
            X,
            weights=weights,
            threshold=threshold,
        )

    def predict_weighted_confidence_scores(self, X, weights=None):
        if self.weighted_confidence_majority is None:
            self.weighted_confidence_majority = wcm.WeightedConfidenceMajority(
                self.model_collections
            )

        return self.weighted_confidence_majority.predict_scores(
            X,
            weights=weights,
        )
        
    # ---- Stacked ----

    def fit_stacked(self, X, y, meta_model=None, use_confidence=True):
        self.stacked = st.Stacked(
            base_models=self.model_collections,
            meta_model=meta_model,
            use_confidence=use_confidence,
        )

        self.stacked.fit(X, y)

        return self

    def predict_stacked(self, X):
        if self.stacked is None:
            raise ValueError("Stacked model has not been fitted yet.")

        return self.stacked.predict(X)