import numpy as np

from src.ensemble.ensemble import Ensemble


class DummyConfidenceCollection:
    def __init__(self, outputs):
        self.outputs = outputs

    def predict_individual(self, X):
        return {
            name: values.argmax(axis=1)
            for name, values in self.outputs.items()
        }

    def predict_confidence(self, X):
        return self.outputs


def accuracy_score(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def test_fit_weighted_confidence_majority_optuna_cv_returns_normalized_weights():
    outputs = {
        "model_a": np.array([
            [0.90, 0.10],
            [0.15, 0.85],
            [0.80, 0.20],
            [0.10, 0.90],
            [0.75, 0.25],
            [0.20, 0.80],
        ]),
        "model_b": np.array([
            [0.60, 0.40],
            [0.35, 0.65],
            [0.55, 0.45],
            [0.25, 0.75],
            [0.70, 0.30],
            [0.30, 0.70],
        ]),
    }
    y = np.array([0, 1, 0, 1, 0, 1])
    ensemble = Ensemble([DummyConfidenceCollection(outputs)])

    weights, threshold, score, model_names = (
        ensemble.fit_weighted_confidence_majority_optuna_cv(
            X=[f"text {i}" for i in range(len(y))],
            y=y,
            score_func=accuracy_score,
            cv=3,
            n_trials=5,
            random_state=7,
            thresholds=[None, 0.6],
        )
    )

    assert len(weights) == 2
    assert np.isclose(weights.sum(), 1.0)
    assert all(weight >= 0 for weight in weights)
    assert threshold in {None, 0.6}
    assert 0.0 <= score <= 1.0
    assert model_names == ["model_a", "model_b"]
