import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from src.model.base_model_collection import BaseModelCollection


class GamingModelCollection(BaseModelCollection):
    def __init__(self, model_joblibs: list[str]):
        self.classifiers = self._set_classifiers(model_joblibs)

    def _set_classifiers(self, model_joblibs):
        classifiers = {}

        for path in model_joblibs:
            loaded = joblib.load(path)

            if isinstance(loaded, list):
                result_dicts = loaded
            else:
                result_dicts = [loaded]

            for result_dict in result_dicts:
                for dataset_name, value in result_dict.items():

                    if "pipes" not in value:
                        continue

                    pipes = value["pipes"]

                    if isinstance(pipes, dict):
                        for model_label, pipe in pipes.items():
                            clean_model_label = model_label.replace(" ", "_")
                            model_name = f"{dataset_name}_{clean_model_label}"
                            classifiers[model_name] = pipe

                    elif isinstance(pipes, list):
                        for pipe_idx, pipe in enumerate(pipes):
                            clf = pipe.named_steps["clf"]
                            model_label = type(clf).__name__
                            model_name = f"{dataset_name}_{model_label}_{pipe_idx}"
                            classifiers[model_name] = pipe

                    else:
                        raise ValueError(
                            f"Unsupported pipes type for {dataset_name}: {type(pipes)}"
                        )

        return classifiers

    def _normalize_texts(self, X):
        """
        Ensure X is raw text in a format sklearn pipelines can consume.
        """
        if isinstance(X, str):
            return [X]

        if isinstance(X, pd.Series):
            return X.fillna("").astype(str)

        if isinstance(X, np.ndarray):
            return pd.Series(X).fillna("").astype(str)

        return pd.Series(X).fillna("").astype(str)

    def predict_individual(self, X):
        """
        Return individual hard predictions from each fitted pipeline.

        X should be raw text, not TF-IDF features.
        """
        X = self._normalize_texts(X)

        predictions = {}

        for model_name, pipe in self.classifiers.items():
            predictions[model_name] = pipe.predict(X)

        return predictions

    def predict_confidence(self, X):
        """
        Return class confidence scores from each fitted pipeline.

        Binary:
            returns shape (n_samples, 2) when possible

        Multiclass:
            returns shape (n_samples, n_classes)
        """
        X = self._normalize_texts(X)

        confidences = {}

        for model_name, pipe in self.classifiers.items():

            if hasattr(pipe, "predict_proba"):
                confs = pipe.predict_proba(X)

            elif hasattr(pipe, "decision_function"):
                raw_scores = pipe.decision_function(X)

                # Binary LinearSVC gives shape (n_samples,)
                if raw_scores.ndim == 1:
                    toxic_scores = expit(raw_scores)
                    confs = np.column_stack([
                        1 - toxic_scores,
                        toxic_scores,
                    ])

                # Multiclass LinearSVC gives shape (n_samples, n_classes)
                else:
                    confs = softmax(raw_scores, axis=1)

            else:
                preds = pipe.predict(X)
                n_classes = len(np.unique(preds))
                confs = np.zeros((len(preds), n_classes))
                confs[np.arange(len(preds)), preds] = 1.0

            confidences[model_name] = confs

        return confidences