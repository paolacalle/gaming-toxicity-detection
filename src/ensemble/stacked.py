import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

# out of fold predictions for meta-model training
# base models are retrained on the full dataset after meta-model training
# meaning the meta-model is always trained on predictions from base models that did not
# train on the same validation fold, ensuring no data leakage

class Stacked:
    def __init__(self, base_models, meta_model, n_folds=5, random_state=42):
        if not base_models:
            raise ValueError("At least one base model is required.")

        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state

        self.fitted_base_models = []
        self.fitted_meta_model = None

    def fit(self, X, y):
        """
        Fits a stacked ensemble using out-of-fold predictions.

        The meta-model is trained on predictions from base models that did not
        train on the same validation fold.
        """
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        X = np.asarray(X)
        y = np.asarray(y)

        meta_features = np.zeros((len(X), len(self.base_models)))

        for model_idx, model in enumerate(self.base_models):
            for train_idx, val_idx in skf.split(X, y):
                model_clone = clone(model)

                model_clone.fit(X[train_idx], y[train_idx])

                meta_features[val_idx, model_idx] = model_clone.predict(X[val_idx])

        self.fitted_meta_model = clone(self.meta_model)
        self.fitted_meta_model.fit(meta_features, y)

        self.fitted_base_models = []

        for model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.fitted_base_models.append(model_clone)

        return self

    def predict(self, X):
        """
        Predicts using the stacked ensemble.
        """
        if self.fitted_meta_model is None:
            raise ValueError("The stacked model must be fitted before prediction.")

        meta_features = self._create_meta_features(X)

        return self.fitted_meta_model.predict(meta_features)

    def _create_meta_features(self, X):
        """
        Creates meta-features using the fitted base models.
        """
        if not self.fitted_base_models:
            raise ValueError("Base models must be fitted before creating meta-features.")

        meta_features = []

        for model in self.fitted_base_models:
            predictions = model.predict(X)
            meta_features.append(predictions)

        return np.column_stack(meta_features)