import joblib
from sklearn.naive_bayes import ComplementNB


class SocialMediaClassifier:
    def __init__(
        self,
        model_joblibs: list[str],
        scaler_path: str,
        nb_tfidf_path: str,
    ):
        self.classifiers = {
            path: joblib.load(path)
            for path in model_joblibs
        }

        self.scaler = joblib.load(scaler_path)
        self.nb_tfidf = joblib.load(nb_tfidf_path)

    def _get_features(self, X):
        """
        Convert raw text into:
        - TF-IDF features for ComplementNB
        - scaled TF-IDF features for LinearSVC / LogisticRegression
        """
        X_nb = self.nb_tfidf.transform(X)
        X_scaled = self.scaler.transform(X_nb)

        return X_nb, X_scaled

    def predict_individual(self, X):
        """
        Return individual hard predictions from each model.
        """
        X_nb, X_scaled = self._get_features(X)

        predictions = {}

        for model_path, clf in self.classifiers.items():
            model_name = model_path.split("/")[-1]

            if isinstance(clf, ComplementNB) or "ComplementNB" in model_name:
                X_use = X_nb
            else:
                X_use = X_scaled

            predictions[model_name] = clf.predict(X_use)

        return predictions

    def predict_confidence(self, X):
        """
        Return toxic-class confidence scores for each model.
        Uses predict_proba when available.
        Uses sigmoid(decision_function) for LinearSVC-like models.
        """
        from scipy.special import expit

        X_nb, X_scaled = self._get_features(X)

        confidences = {}

        for model_path, clf in self.classifiers.items():
            model_name = model_path.split("/")[-1]

            if isinstance(clf, ComplementNB) or "ComplementNB" in model_name:
                X_use = X_nb
            else:
                X_use = X_scaled

            if hasattr(clf, "predict_proba"):
                toxic_score = clf.predict_proba(X_use)[:, 1]

            elif hasattr(clf, "decision_function"):
                raw_score = clf.decision_function(X_use)
                toxic_score = expit(raw_score)

            else:
                toxic_score = clf.predict(X_use)

            confidences[model_name] = toxic_score

        return confidences
