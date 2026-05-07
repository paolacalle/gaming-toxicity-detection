"""
utils.py — Cross-cutting utilities: seeding, device, logging, metrics,
           class-weight computation, and plotting.

Public API
----------
get_logger(name)                                → logging.Logger
set_seed(seed)                                  → None
get_device()                                    → torch.device
get_class_weights(labels, n_classes)            → torch.Tensor
compute_metrics_binary(eval_pred)               → dict   [HF Trainer callback]
make_compute_metrics_mc(n_classes)              → callable
compute_all_metrics(y_true, y_pred, task_type)  → dict   [for saving / printing]
save_metrics(metrics, path)                     → None
load_metrics(path)                              → dict
plot_training_history(log_history, ...)         → None   [saves PNG to disk]
plot_confusion_matrix(y_true, y_pred, ...)      → None   [saves PNG to disk]
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# NOTE: matplotlib and seaborn are imported lazily inside the plotting
# functions below (plot_training_history, plot_confusion_matrix).
#
# Root cause (Windows-specific): importing matplotlib.pyplot after
# transformers.Trainer (which loads Accelerate + CUDA DLLs) corrupts
# PyArrow's memory allocator so that any subsequent pd.read_parquet() call
# crashes the process with STATUS_ACCESS_VIOLATION (0xC0000005 / exit 3221225477).
# Deferring both imports to the plotting calls — which always happen after
# all data loading and training are complete — avoids the conflict entirely.
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that writes to stderr with a timestamped format.

    Re-using this function throughout the project ensures that every module
    produces identically formatted log lines that are easy to distinguish
    by caller name.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for fully reproducible training runs.

    Sets seeds for Python's ``random``, NumPy, and PyTorch (both CPU and
    all CUDA devices if available).  Also pins ``PYTHONHASHSEED`` so that
    dict/set iteration order is deterministic in CPython 3.6+.

    Parameters
    ----------
    seed : int
        Any non-negative integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available compute device.

    Returns ``cuda`` if at least one NVIDIA GPU is visible to PyTorch,
    otherwise falls back to ``cpu``.

    Returns
    -------
    torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------

def get_class_weights(labels: list, n_classes: int) -> torch.Tensor:
    """
    Compute balanced inverse-frequency class weights.

    Uses scikit-learn's ``"balanced"`` strategy, which sets the weight for
    class *k* to ``n_samples / (n_classes * count_k)``.  This makes the
    model penalise mistakes on rare (toxic) classes proportionally more
    than mistakes on the dominant (non-toxic) class.

    Parameters
    ----------
    labels : list[int]
        Integer labels from the **training** split only.  Never pass val or
        test labels here — that would let test-set statistics influence
        training.
    n_classes : int
        Total number of classes, including any that may be absent from
        the label list (e.g. very rare WOT level-5 examples in a small split).

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(n_classes,)`` on the CPU.  Move to the
        target device after calling this function.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=np.array(labels),
    )
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# HuggingFace Trainer metric callbacks
# ---------------------------------------------------------------------------

def compute_metrics_binary(eval_pred) -> dict:
    """
    Metric callback for the HuggingFace ``Trainer`` in binary mode.

    Called after each validation pass.  The returned values are used for
    early-stopping (to monitor ``"f1"``) and for logging to stdout / W&B.
    These are **validation-set** figures; they are not the final test
    numbers reported in the evaluation summary.

    Parameters
    ----------
    eval_pred : transformers.EvalPrediction
        Named tuple with fields ``predictions`` (raw logits) and
        ``label_ids`` (integer ground-truth labels).

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``f1``, ``f1_macro``, ``precision``, ``recall``.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, average="binary",   zero_division=0)),
        "f1_macro":  float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="binary",    zero_division=0)),
    }


def make_compute_metrics_mc(n_classes: int) -> Callable:
    """
    Factory that returns a multi-class metric callback for HuggingFace Trainer.

    Macro F1 weights every class equally regardless of support, making it
    sensitive to the model's performance on rare high-toxicity classes.
    Weighted F1 accounts for class size, giving a better picture of
    overall dataset-level quality.

    Parameters
    ----------
    n_classes : int
        Number of classes.  Passed explicitly to prevent ``f1_score`` from
        silently ignoring classes that have no predictions.

    Returns
    -------
    callable
        Function with signature ``(eval_pred) -> dict[str, float]``.
    """
    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy":    float(accuracy_score(labels, preds)),
            "f1_macro":    float(
                f1_score(labels, preds, average="macro",
                         labels=list(range(n_classes)), zero_division=0)
            ),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Final evaluation metrics (for saving and printing)
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: list,
    y_pred: list,
    task_type: str = "binary",
    label_names: Optional[list] = None,
) -> dict:
    """
    Compute a comprehensive set of evaluation metrics.

    Used after training is complete to produce the numbers stored in
    ``eval_metrics.json`` and printed in the final summary.  This function
    always operates on the held-out **test set**.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth labels.
    y_pred : list[int]
        Model predictions (argmax of logits).
    task_type : str
        ``"binary"`` or ``"multiclass"``.
    label_names : list[str], optional
        Human-readable class names used in ``classification_report``.
        Defaults to ``["Non-Toxic", "Toxic"]`` for binary and
        ``["Class-0", ...]`` for multi-class.

    Returns
    -------
    dict[str, float]
        A flat dict with keys prefixed to indicate their averaging strategy
        (e.g. ``"f1_binary"``, ``"f1_macro"``, ``"precision_macro"``).
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n_classes = max(max(y_true), max(y_pred)) + 1

    if label_names is None:
        if task_type == "binary":
            label_names = ["Non-Toxic", "Toxic"]
        else:
            label_names = [f"Class-{i}" for i in range(n_classes)]

    metrics: dict = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average="macro",       zero_division=0)),
    }

    # Additional binary-specific scalars (more informative than macro
    # averages when there are only two classes)
    if task_type == "binary":
        metrics["f1_binary"]        = float(f1_score(y_true, y_pred, average="binary",   zero_division=0))
        metrics["precision_binary"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall_binary"]    = float(recall_score(y_true, y_pred, zero_division=0))

    # Full per-class breakdown as a nested dict (mirrors sklearn's report)
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = report

    return metrics


# ---------------------------------------------------------------------------
# Metric I/O
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Serialise a metrics dict to a JSON file.

    Parent directories are created automatically.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_all_metrics`.
    output_path : str
        Destination file path (e.g. ``"outputs/wot_binary/eval_metrics.json"``).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def load_metrics(path: str) -> dict:
    """
    Load a previously saved metrics JSON file.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(
    log_history: list,
    title: str,
    save_path: str,
    metric_key: str = "eval_f1",
) -> None:
    """
    Save a two-panel training-curve figure to *save_path*.

    Left panel  — training loss over steps (from the ``"loss"`` entries in
                  ``log_history``).
    Right panel — validation metric over steps (from ``metric_key``  entries).

    The validation metric shown here is the **early-stopping signal** (val
    set), not the final test-set metric.

    Parameters
    ----------
    log_history : list[dict]
        ``trainer.state.log_history`` after training completes.
    title : str
        Figure title prefix (e.g. ``"WOT Binary"``).
    save_path : str
        Destination PNG file path.
    metric_key : str
        Key used to extract the validation metric from ``log_history``
        (e.g. ``"eval_f1"`` for binary, ``"eval_f1_macro"`` for multi-class).
    """
    # Deferred import — see module-level comment about Windows crash.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_steps, train_losses = [], []
    val_steps, val_values = [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if metric_key in entry:
            val_steps.append(entry["step"])
            val_values.append(entry[metric_key])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_steps, train_losses, marker="o", markersize=3, linewidth=1)
    axes[0].set_title(f"{title} — Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")

    ylabel = metric_key.replace("eval_", "").replace("_", " ").title()
    axes[1].plot(val_steps, val_values, marker="s", color="darkorange",
                 markersize=5, linewidth=1.5)
    axes[1].set_title(f"{title} — Val {ylabel} (early-stop signal)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(ylabel)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    label_names: list,
    title: str,
    save_path: str,
) -> None:
    """
    Save a labelled confusion-matrix heatmap to *save_path*.

    The matrix always shows **test-set** predictions; it is never called
    with validation data.

    Parameters
    ----------
    y_true : list[int]
        True labels.
    y_pred : list[int]
        Predicted labels.
    label_names : list[str]
        Human-readable class names.  The length must equal the number of
        distinct classes in ``y_true`` ∪ ``y_pred``.
    title : str
        Figure title.
    save_path : str
        Destination PNG file path.
    """
    # Deferred imports — see module-level comment about Windows crash.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    fig_w = max(5, n * 1.5)
    fig_h = max(4, n * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
