import numpy as np
from sklearn.linear_model import LogisticRegression


class Stacked:
    def __init__(
        self,
        base_models,
        meta_model=None,
        use_confidence=True,
    ):
        """
        Stacked ensemble using already-fitted model collections.

        Each base model collection must implement:
            predict_individual(X) -> dict[str, np.ndarray]
            predict_confidence(X) -> dict[str, np.ndarray]

        The base models are NOT refit. They only generate meta-features.
        Then a logistic regression model is fit on those features.
        """
        if not base_models:
            raise ValueError("At least one base model collection is required.")

        self.base_models = base_models
        self.use_confidence = use_confidence

        self.meta_model = meta_model or LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )

        self.fitted_meta_model = None
        self.feature_names_ = None

    def fit(self, X, y):
        """
        Create meta-features from base model outputs, then fit logistic regression.
        """
        X_meta = self._create_meta_features(X)
        y = np.asarray(y)

        self.fitted_meta_model = self.meta_model
        self.fitted_meta_model.fit(X_meta, y)

        return self

    def predict(self, X):
        """
        Predict using the logistic regression meta-model.
        """
        if self.fitted_meta_model is None:
            raise ValueError("Stacked model has not been fitted yet.")

        X_meta = self._create_meta_features(X)

        return self.fitted_meta_model.predict(X_meta)

    def predict_proba(self, X):
        """
        Return probabilities from the logistic regression meta-model.
        """
        if self.fitted_meta_model is None:
            raise ValueError("Stacked model has not been fitted yet.")

        X_meta = self._create_meta_features(X)

        return self.fitted_meta_model.predict_proba(X_meta)

    def _create_meta_features(self, X):
        """
        Build one feature column per base model.

        If use_confidence=True:
            feature = toxic confidence score from each model.

        If use_confidence=False:
            feature = hard prediction from each model.
        """
        meta_feature_dict = {}

        for collection in self.base_models:
            if self.use_confidence:
                outputs = collection.predict_confidence(X)
            else:
                outputs = collection.predict_individual(X)

            for model_name, values in outputs.items():
                if model_name in meta_feature_dict:
                    raise ValueError(
                        f"Duplicate model name found: {model_name}. "
                        "Model names must be unique across collections."
                    )

                meta_feature_dict[model_name] = np.asarray(values)

        self.feature_names_ = list(meta_feature_dict.keys())

        return np.column_stack(
            [meta_feature_dict[name] for name in self.feature_names_]
        )