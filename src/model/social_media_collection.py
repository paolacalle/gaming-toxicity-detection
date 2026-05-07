from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from src.preprocess.social_media_text_preprocessor import SocialMediaTextPreprocessor
from src.model.base_model_collection import BaseModelCollection


class SocialMediaModelCollection(BaseModelCollection):
    def __init__(
        self,
        model_joblibs: list[str | Path],
        scaler_path: str | Path,
        nb_tfidf_path: str | Path,
        preprocessor: SocialMediaTextPreprocessor | None = None,
        output_mode: str = "multioutput",
    ):
        """
        Parameters
        ----------
        model_joblibs:
            Paths to saved social media models.

        scaler_path:
            Fitted scaler used for LR/SVC models.

        nb_tfidf_path:
            Fitted TF-IDF vectorizer.

        preprocessor:
            Text preprocessor.

        output_mode:
            "multioutput" -> return multi-output predictions as-is.
            "single_label" -> convert multi-output predictions into one class label.
        """
        self.classifiers = {
            Path(path): joblib.load(path)
            for path in model_joblibs
        }

        self.scaler = joblib.load(scaler_path)
        self.nb_tfidf = joblib.load(nb_tfidf_path)
        self.preprocessor = preprocessor or SocialMediaTextPreprocessor()
        self.output_mode = output_mode

    def _get_features(self, X):
        """
        Convert raw text into:
        - TF-IDF features for ComplementNB
        - scaled TF-IDF features for LinearSVC / LogisticRegression
        """
        X = self.preprocessor.preprocess_series(X)

        X_nb = self.nb_tfidf.transform(X)
        X_scaled = self.scaler.transform(X_nb)

        return X_nb, X_scaled

    def _uses_nb_features(self, model_name: str) -> bool:
        """
        Decide whether this model should use unscaled TF-IDF.

        Your NB model path is:
            social_media_multioutput(nb)_pipeline.joblib
        """
        model_name_lower = model_name.lower()

        return (
            "nb" in model_name_lower
            or "complementnb" in model_name_lower
            or "complement_nb" in model_name_lower
        )

    def _clean_model_name(self, model_path: Path) -> str:
        return model_path.stem

    def _multioutput_to_single_label(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Convert multi-output binary predictions into a single class label.

        Example for 2 output columns:
            [0, 0] -> 0
            [1, 0] -> 1
            [0, 1] -> 2
            [1, 1] -> 2  # chooses highest active class

        This assumes columns represent increasing toxicity levels.
        """
        y_pred = np.asarray(y_pred)

        if y_pred.ndim == 1:
            return y_pred

        single_labels = np.zeros(y_pred.shape[0], dtype=int)

        for row_idx, row in enumerate(y_pred):
            active = np.where(row == 1)[0]

            if len(active) == 0:
                single_labels[row_idx] = 0
            else:
                single_labels[row_idx] = active.max() + 1

        return single_labels

    def _multioutput_proba_to_class_confidence(self, probas) -> np.ndarray:
        """
        Convert MultiOutputClassifier predict_proba output into class-confidence matrix.

        MultiOutputClassifier returns:
            list[array], where each array has shape (n_samples, 2)

        For 2 output labels, returns:
            shape (n_samples, 3)
            columns = [class_0_conf, class_1_conf, class_2_conf]

        class_0_conf is estimated as 1 - max positive-label confidence.
        """
        if not isinstance(probas, list):
            return np.asarray(probas)

        positive_probs = np.column_stack([
            output_proba[:, 1]
            for output_proba in probas
        ])
        # shape: (n_samples, n_outputs)

        none_prob = 1 - positive_probs.max(axis=1)

        class_conf = np.column_stack([
            none_prob,
            positive_probs,
        ])

        # Avoid negative values just in case probabilities are weird
        class_conf = np.clip(class_conf, 0, 1)

        # Normalize rows so they behave like class probabilities
        row_sums = class_conf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1

        return class_conf / row_sums

    def _multioutput_decision_to_class_confidence(self, raw_scores) -> np.ndarray:
        """
        Convert MultiOutputClassifier decision_function output into class-confidence matrix.

        For LinearSVC wrapped in MultiOutputClassifier, decision_function may return:
            list[array] or array-like of binary margins.

        Converts margins to sigmoid confidence, then builds:
            [none_conf, output_1_conf, output_2_conf, ...]
        """
        if isinstance(raw_scores, list):
            positive_scores = np.column_stack([
                expit(scores)
                for scores in raw_scores
            ])
        else:
            raw_scores = np.asarray(raw_scores)

            if raw_scores.ndim == 1:
                positive_scores = expit(raw_scores).reshape(-1, 1)
            else:
                positive_scores = expit(raw_scores)

        none_prob = 1 - positive_scores.max(axis=1)

        class_conf = np.column_stack([
            none_prob,
            positive_scores,
        ])

        class_conf = np.clip(class_conf, 0, 1)

        row_sums = class_conf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1

        return class_conf / row_sums

    def predict_individual(self, X) -> dict[str, np.ndarray]:
        """
        Return individual hard predictions from each social media model.
        """
        X_nb, X_scaled = self._get_features(X)

        predictions = {}

        for model_path, model in self.classifiers.items():
            model_name = self._clean_model_name(model_path)

            X_use = X_nb if self._uses_nb_features(model_name) else X_scaled

            y_pred = model.predict(X_use)

            if self.output_mode == "single_label":
                y_pred = self._multioutput_to_single_label(y_pred)

            predictions[model_name] = y_pred

        return predictions

    def predict_confidence(self, X) -> dict[str, np.ndarray]:
        """
        Return confidence scores from each social media model.

        For multi-output models:
            returns class-confidence matrix:
                shape (n_samples, n_classes)

        Example for two binary outputs:
            columns = [non-toxic, mild, severe]
        """
        X_nb, X_scaled = self._get_features(X)

        confidences = {}

        for model_path, model in self.classifiers.items():
            model_name = self._clean_model_name(model_path)

            X_use = X_nb if self._uses_nb_features(model_name) else X_scaled

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_use)
                confs = self._multioutput_proba_to_class_confidence(probas)

            elif hasattr(model, "decision_function"):
                raw_scores = model.decision_function(X_use)
                confs = self._multioutput_decision_to_class_confidence(raw_scores)

            else:
                y_pred = model.predict(X_use)

                if self.output_mode == "single_label":
                    y_pred = self._multioutput_to_single_label(y_pred)

                # fallback: one-hot hard confidence
                n_classes = int(np.max(y_pred)) + 1
                confs = np.zeros((len(y_pred), n_classes))
                confs[np.arange(len(y_pred)), y_pred] = 1.0

            confidences[model_name] = confs

        return confidences