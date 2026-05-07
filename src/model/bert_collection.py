from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.model.base_model_collection import BaseModelCollection

class BertToxicityModelCollection(BaseModelCollection):
    """
    Wrapper for BERT toxicity models from:
    https://huggingface.co/jforward/bert-toxicity

    Supports loading one or more models from subfolders.
    """

    MODEL_SUBFOLDERS = {
        "dota_binary": "dota_binary_model",
        "dota_multi": "dota_multi_model",
        "jigsaw_binary": "jigsaw_binary_model",
        "jigsaw_multi": "jigsaw_multi_model",
        "wot_binary": "wot_binary_model",
        "wot_multi": "wot_multi_model",
    }

    def __init__(
        self,
        model_names: list[str],
        repo_id: str = "jforward/bert-toxicity",
        max_length: int = 128,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self._validate_model_names(model_names)

        self.model_names = model_names
        self.repo_id = repo_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizers = {}
        self.models = {}

        for model_name in self.model_names:
            subfolder = self.MODEL_SUBFOLDERS[model_name]

            print(f"Loading {model_name} from {self.repo_id}/{subfolder}...")

            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                self.repo_id,
                subfolder=subfolder,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.repo_id,
                subfolder=subfolder,
            )

            model.to(self.device)
            model.eval()

            self.models[model_name] = model

    def _validate_model_names(self, model_names: list[str]) -> None:
        for model_name in model_names:
            if model_name not in self.MODEL_SUBFOLDERS:
                raise ValueError(
                    f"Unknown model_name: {model_name}. "
                    f"Expected one of {list(self.MODEL_SUBFOLDERS.keys())}."
                )

    def _normalize_texts(self, texts) -> list[str]:
        if isinstance(texts, str):
            return [texts]

        if isinstance(texts, pd.Series):
            return texts.fillna("").astype(str).tolist()

        if isinstance(texts, np.ndarray):
            return [str(x) if x is not None else "" for x in texts]

        return [str(x) if x is not None else "" for x in texts]

    def predict_logits(self, texts) -> dict[str, np.ndarray]:
        """
        Return raw logits for each loaded model.

        Returns
        -------
        dict[str, np.ndarray]
            model_name -> logits array of shape (n_samples, n_classes)
        """
        texts = self._normalize_texts(texts)

        all_logits = {model_name: [] for model_name in self.model_names}

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]

            for model_name in self.model_names:
                encoded = self.tokenizers[model_name](
                    batch_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = self.models[model_name](**encoded)
                    logits = outputs.logits

                all_logits[model_name].append(logits.cpu().numpy())

        return {
            model_name: np.vstack(logits_list)
            for model_name, logits_list in all_logits.items()
        }

    def predict_proba(self, texts) -> dict[str, np.ndarray]:
        """
        Return softmax probabilities for each loaded model.

        Returns
        -------
        dict[str, np.ndarray]
            model_name -> probability array of shape (n_samples, n_classes)
        """
        logits = self.predict_logits(texts)

        probs = {}

        for model_name, model_logits in logits.items():
            exp_logits = np.exp(
                model_logits - np.max(model_logits, axis=1, keepdims=True)
            )
            probs[model_name] = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return probs

    def predict_individual(self, texts) -> dict[str, np.ndarray]:
        """
        Return hard predictions for each loaded model.
        """
        probs = self.predict_proba(texts) # model_name : (n_samples, n_classes) probabilities

        # model_name : predicted class (n_samples,)
        return {
            model_name: model_probs.argmax(axis=1)
            for model_name, model_probs in probs.items()
        }

    def predict_confidence(self, texts) -> dict[str, np.ndarray]:
        """
        Return class probability/confidence scores for each model.

        For binary models:
            returns shape (n_samples, 2)

        For multiclass models:
            returns shape (n_samples, n_classes)

        Returns
        -------
        dict[str, np.ndarray]
            model_name -> probability matrix of shape (n_samples, n_classes)
        """
        probs = self.predict_proba(texts)

        confidences = {}

        for model_name, model_probs in probs.items():
            if model_probs.ndim != 2:
                raise ValueError(
                    f"{model_name} predict_proba should return a 2D array, "
                    f"got shape {model_probs.shape}."
                )

            confidences[model_name] = model_probs

        return confidences