import numpy as np

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
)

class ClassificationMetrics:
    def __init__(
        self,
        uncertain_label: int = -1,
        min_coverage: float = 0.80,
        positive_label: int = 1,
        negative_label: int = 0,
    ):
        self.uncertain_label = uncertain_label
        self.min_coverage = min_coverage
        self.positive_label = positive_label
        self.negative_label = negative_label

    def safe_confusion_rates(self, y_true, y_pred) -> dict[str, float]:
        """
        Compute binary confusion-matrix rates.

        Assumes:
            negative_label = 0
            positive_label = 1

        Returns:
            FPR, FNR, TPR, TNR
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[self.negative_label, self.positive_label],
        ).ravel()

        return {
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "TNR": tn / (tn + fp) if (tn + fp) > 0 else 0,
        }

    def score(self, y_true, y_pred) -> float:
        """
        Score function for threshold tuning.

        If predictions include uncertain_label, only covered predictions are scored.
        If coverage is below min_coverage, returns -inf.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        covered = y_pred != self.uncertain_label
        coverage = covered.mean()

        if coverage < self.min_coverage:
            return -np.inf

        return f1_score(
            y_true[covered],
            y_pred[covered],
            average="macro",
            zero_division=0,
        )

    def precision_from_rates(self, y_true, y_pred) -> float:
        """
        Your original precision-like formula using TPR and FPR.

        Note: this is not standard precision.
        Standard precision is TP / (TP + FP).
        """
        rates = self.safe_confusion_rates(y_true, y_pred)

        denominator = rates["TPR"] + rates["FPR"]

        if denominator == 0:
            return 0

        return rates["TPR"] / denominator

    def metrics(self, y_true, y_pred) -> dict[str, float]:
        """
        Compute prediction metrics.

        Handles uncertain_label by reporting:
            - coverage
            - metrics on covered predictions only
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        covered = y_pred != self.uncertain_label
        coverage = covered.mean()

        y_true_eval = y_true[covered]
        y_pred_eval = y_pred[covered]

        if len(y_true_eval) == 0:
            return {
                "CV Macro F1": 0,
                "CV Weighted F1": 0,
                "Accuracy": 0,
                "Precision": 0,
                "Coverage": 0,
                "FPR": 0,
                "FNR": 0,
                "TPR": 0,
                "TNR": 0,
            }

        result = {
            "CV Macro F1": f1_score(
                y_true_eval,
                y_pred_eval,
                average="macro",
                zero_division=0,
            ),
            "CV Weighted F1": f1_score(
                y_true_eval,
                y_pred_eval,
                average="weighted",
                zero_division=0,
            ),
            "Accuracy": accuracy_score(y_true_eval, y_pred_eval),
            "Coverage": coverage,
        }

        # Only add binary confusion rates when this is a binary task.
        unique_labels = set(np.unique(y_true_eval)) | set(np.unique(y_pred_eval))

        if unique_labels.issubset({self.negative_label, self.positive_label}):
            result["Precision"] = precision_score(
                y_true_eval,
                y_pred_eval,
                pos_label=self.positive_label,
                zero_division=0,
            )
            result.update(self.safe_confusion_rates(y_true_eval, y_pred_eval))
        else:
            # For multiclass, standard binary FPR/FNR/TPR/TNR do not directly apply.
            result["Precision"] = precision_score(
                y_true_eval,
                y_pred_eval,
                average="macro",
                zero_division=0,
            )

        return result