"""
model.py — BERT model construction, weighted trainer, and save/load utilities.

Public API
----------
build_tokenizer(model_name)               → BertTokenizer
build_model(model_name, num_labels)       → BertForSequenceClassification
save_model(model, tokenizer, output_dir)  → None
load_model(model_dir)                     → (model, tokenizer)
WeightedTrainer                           → transformers.Trainer subclass
"""

import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizer:
    """
    Load a BERT tokenizer from the HuggingFace hub or a local directory.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"bert-base-uncased"``) or a
        path to a locally saved tokenizer directory.

    Returns
    -------
    BertTokenizer
    """
    return BertTokenizer.from_pretrained(model_name)


def build_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
) -> BertForSequenceClassification:
    """
    Load a pre-trained BERT encoder with a fresh classification head.

    The classification head is a single linear layer whose output size is
    automatically set to ``num_labels``.  It is randomly initialised and
    will be trained during fine-tuning; the encoder weights are loaded from
    the checkpoint.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or local directory.
    num_labels : int
        Number of output classes.  Use 2 for binary classification,
        or the number of ordinal levels for multi-class.

    Returns
    -------
    BertForSequenceClassification
    """
    return BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    output_dir: str,
) -> None:
    """
    Save a fine-tuned model and its tokenizer to *output_dir*.

    Both the model weights (``pytorch_model.bin`` or safetensors shards)
    and the tokenizer vocabulary/config files are written so that the
    directory is fully self-contained for later loading.

    Parameters
    ----------
    model : BertForSequenceClassification
    tokenizer : BertTokenizer
    output_dir : str
        Destination directory (created if it does not exist).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_model(
    model_dir: str,
) -> tuple:
    """
    Load a fine-tuned model and its tokenizer from a saved directory.

    The directory must have been written by :func:`save_model` or by
    HuggingFace's ``Trainer`` (which produces the same layout).

    Parameters
    ----------
    model_dir : str
        Path to the directory that contains ``config.json``,
        model weights, and tokenizer files.  Relative paths (including ones
        starting with ``./`` or ``.\\``) are resolved to an absolute path
        before being passed to HuggingFace so that the library does not
        mistake a relative local path for a Hub repo ID.

    Returns
    -------
    tuple[BertForSequenceClassification, BertTokenizer]
    """
    from pathlib import Path as _Path
    # Resolve to an absolute path so HuggingFace always sees a local directory,
    # never a relative path that could be misidentified as a Hub repo ID.
    abs_dir = str(_Path(model_dir).resolve())
    model     = BertForSequenceClassification.from_pretrained(abs_dir)
    tokenizer = BertTokenizer.from_pretrained(abs_dir)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    HuggingFace ``Trainer`` subclass that uses class-weighted cross-entropy.

    Motivation
    ----------
    All three toxicity datasets are class-imbalanced (non-toxic messages
    vastly outnumber toxic ones).  Without reweighting, a model that always
    predicts "non-toxic" achieves high accuracy while providing zero signal
    for the minority class.

    Class-weighted cross-entropy multiplies the per-sample loss by an
    inverse-frequency weight so that misclassifying a rare (toxic) sample
    costs proportionally more than misclassifying a common (non-toxic) one.

    Usage
    -----
    Pass the output of :func:`utils.get_class_weights` as ``class_weights``
    and specify ``device`` explicitly so the weight tensor lives on the
    same device as the model::

        trainer = WeightedTrainer(
            class_weights = get_class_weights(train_labels, n_classes=2),
            device        = torch.device("cuda"),
            model         = model,
            args          = training_args,
            ...
        )

    Parameters
    ----------
    class_weights : torch.Tensor
        1-D float tensor of length ``num_labels``, one weight per class.
        Computed by :func:`utils.get_class_weights`.
    device : torch.device
        Device to place the weight tensor on (must match the model device).
    *args, **kwargs
        Forwarded verbatim to ``Trainer.__init__``.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Move the weight tensor to the same device as the model so that
        # the loss computation does not trigger a device-mismatch error.
        self.class_weights = class_weights.to(device)

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Override the default cross-entropy loss with a class-weighted version.

        The ``**kwargs`` absorbs any extra arguments that newer versions of
        HuggingFace Transformers may pass to ``compute_loss`` (e.g.
        ``num_items_in_batch`` introduced in 4.46).
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
