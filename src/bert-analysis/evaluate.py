"""
evaluate.py — CLI entry point for evaluating a saved BERT toxicity model.

Loads a model directory produced by train.py, runs it on a held-out test
set, and writes a full evaluation report.

Usage
-----
Typical workflow (model + test data from a prior train.py run):

    python evaluate.py \\
        --model_dir  outputs/wot_binary/model \\
        --test_data  processed_data/wot/wot_test.parquet \\
        --output_dir outputs/wot_binary/eval

Override task settings when the model_dir does not contain a config:

    python evaluate.py \\
        --model_dir   outputs/my_model/model  \\
        --test_data   my_test.csv             \\
        --task_type   multiclass              \\
        --num_classes 4                       \\
        --label_names "Non-toxic,Mild,Moderate,Severe" \\
        --output_dir  outputs/my_model/eval

Cross-domain evaluation (model trained on WOT, tested on DOTA):

    python evaluate.py \\
        --model_dir  outputs/wot_binary/model          \\
        --test_data  processed_data/dota/dota_test.parquet \\
        --output_dir outputs/wot_binary/cross_dota

Outputs written to --output_dir:
    eval_metrics.json   — accuracy, F1, precision, recall + per-class report
    confusion_matrix.png — heatmap
    predictions.csv     — aligned (true_label, predicted_label) for inspection
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import Trainer, TrainingArguments

from data import ToxicityDataset, get_texts, infer_num_classes, prepare_labels, load_dataframe
from model import load_model
from utils import (
    compute_all_metrics,
    get_device,
    get_logger,
    plot_confusion_matrix,
    save_metrics,
    set_seed,
)

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Define and parse all command-line arguments for evaluation.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a saved BERT toxicity model on a test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # In-domain eval\n"
            "  python evaluate.py --model_dir outputs/wot_binary/model \\\n"
            "                     --test_data  data/wot_test.parquet   \\\n"
            "                     --output_dir outputs/wot_binary/eval\n\n"
            "  # Cross-domain eval\n"
            "  python evaluate.py --model_dir outputs/wot_binary/model  \\\n"
            "                     --test_data  data/dota_test.parquet   \\\n"
            "                     --output_dir outputs/wot_binary/cross_dota\n"
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument(
        "--model_dir",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Directory containing a fine-tuned model (config.json + weights + "
            "tokenizer files).  Produced by train.py's --output_dir/model/."
        ),
    )
    model_grp.add_argument(
        "--config_path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to training_config.json produced by train.py. "
            "Defaults to <model_dir>/../training_config.json. "
            "Explicitly set this when the config lives elsewhere."
        ),
    )

    # ── Data ──────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("data")
    data_grp.add_argument(
        "--test_data",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the held-out test CSV or parquet. Never used during training.",
    )
    data_grp.add_argument(
        "--text_col",
        type=str,
        default=None,
        help="Text column name (overrides training config).",
    )
    data_grp.add_argument(
        "--label_col",
        type=str,
        default=None,
        help="Label column name (overrides training config).",
    )

    # ── Task — override the training config values if needed ──────────────
    task_grp = parser.add_argument_group("task (override training config)")
    task_grp.add_argument(
        "--task_type",
        type=str,
        choices=["binary", "multiclass"],
        default=None,
        help="Override task type from training config.",
    )
    task_grp.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Override number of classes from training config.",
    )
    task_grp.add_argument(
        "--label_names",
        type=str,
        default=None,
        metavar="NAME,NAME,...",
        help=(
            "Comma-separated class names for the report and confusion matrix. "
            "Overrides training config."
        ),
    )

    # ── Inference settings ────────────────────────────────────────────────
    infer_grp = parser.add_argument_group("inference")
    infer_grp.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Inference batch size (overrides training config default of 32).",
    )
    infer_grp.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max tokenizer sequence length (overrides training config).",
    )

    # ── Output ────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("output")
    out_grp.add_argument(
        "--output_dir",
        type=str,
        default="./eval_output",
        help="Directory to write eval_metrics.json, confusion_matrix.png, predictions.csv.",
    )
    out_grp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for reproducibility of any stochastic post-processing).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_training_config(args: argparse.Namespace) -> dict:
    """
    Load the training_config.json written by train.py, then apply any
    CLI overrides the user has provided.

    If no config file exists the function falls back to sensible defaults
    so that evaluate.py works even when called on externally trained models.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    # Resolve model_dir to an absolute path before deriving sibling paths.
    model_dir_abs = Path(args.model_dir).resolve()
    config_path = args.config_path or (
        model_dir_abs.parent / "training_config.json"
    )

    config: dict = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        logger.info(f"Loaded training config: {config_path}")
    else:
        logger.warning(
            f"training_config.json not found at {config_path}. "
            "Using defaults — you may need to set --task_type and --num_classes manually."
        )

    # CLI args take precedence over the saved config
    if args.task_type:
        config["task_type"] = args.task_type
    if args.num_classes:
        config["num_classes"] = args.num_classes
    if args.label_names:
        config["label_names"] = [n.strip() for n in args.label_names.split(",")]
    if args.text_col:
        config["text_col"] = args.text_col
    if args.label_col:
        config["label_col"] = args.label_col
    if args.max_length:
        config["max_length"] = args.max_length
    if args.batch_size:
        config["batch_size"] = args.batch_size

    # Fill in any remaining gaps with defaults
    config.setdefault("task_type",  "binary")
    config.setdefault("num_classes", 2)
    config.setdefault("max_length",  64)
    config.setdefault("batch_size",  32)
    config.setdefault("text_col",    "message")
    config.setdefault("label_col",   "label")

    return config


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference(model, test_dataset: ToxicityDataset,
                   batch_size: int, device,
                   output_dir: str = None) -> np.ndarray:
    """
    Run the model on *test_dataset* and return raw logits.

    Uses HuggingFace ``Trainer.predict()`` which handles device placement,
    batching, and mixed-precision automatically.

    Parameters
    ----------
    model : BertForSequenceClassification
    test_dataset : ToxicityDataset
    batch_size : int
    device : torch.device
    output_dir : str, optional
        Base output directory for this evaluation run.  A ``_tmp_predict``
        subdirectory is created inside it to hold the HuggingFace Trainer's
        checkpoint artefacts (which are never read back).  When *None* a
        system temporary directory is used instead.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, n_classes)`` — raw logits (not softmax).
    """
    import tempfile

    # Determine where to write the Trainer's throwaway checkpoint files.
    # Using a subdirectory of output_dir keeps everything self-contained;
    # falling back to a real temp dir ensures correctness when output_dir
    # is not provided.
    if output_dir is not None:
        tmp_dir = str(Path(output_dir) / "_tmp_predict")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="bert_eval_")

    # A minimal TrainingArguments is needed just to configure the predict pass.
    predict_args = TrainingArguments(
        output_dir                 = tmp_dir,
        per_device_eval_batch_size = batch_size,
        report_to                  = "none",
        disable_tqdm               = False,
    )
    predictor = Trainer(model=model, args=predict_args)
    predictions = predictor.predict(test_dataset)
    return predictions.predictions   # shape (N, num_classes)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end evaluation pipeline.

    Steps
    -----
    1.  Parse CLI arguments.
    2.  Load the training config (from file + CLI overrides).
    3.  Set random seeds.
    4.  Load the fine-tuned model and tokenizer.
    5.  Load and tokenise the test data.
    6.  Prepare ground-truth labels.
    7.  Run inference with ``Trainer.predict()``.
    8.  Convert logits to predicted class indices.
    9.  Compute and print the full classification report.
    10. Save ``eval_metrics.json``, ``confusion_matrix.png``,
        and ``predictions.csv`` to ``--output_dir``.
    """
    args = parse_args()

    # ── 1-2. Config ───────────────────────────────────────────────────────
    cfg = _load_training_config(args)
    set_seed(args.seed)
    device = get_device()
    logger.info(
        f"Evaluating  model_dir={args.model_dir}  "
        f"task={cfg['task_type']}  classes={cfg['num_classes']}"
    )

    # ── 3. Label names ────────────────────────────────────────────────────
    label_names: list = cfg.get("label_names") or (
        ["Non-Toxic", "Toxic"]
        if cfg["task_type"] == "binary"
        else [f"Class-{i}" for i in range(cfg["num_classes"])]
    )

    # ── 4. Load model ─────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model_dir} …")
    model, tokenizer = load_model(args.model_dir)
    model.eval()

    # ── 5. Load and tokenise test data ────────────────────────────────────
    logger.info(f"Loading test data: {args.test_data}")
    test_df = load_dataframe(args.test_data)
    logger.info(f"Test set: {len(test_df):,} rows")

    test_texts  = get_texts(test_df, cfg["text_col"])
    test_labels = prepare_labels(test_df, cfg["label_col"], cfg["task_type"])

    # Validate that the inferred number of classes matches the model head
    inferred_n = infer_num_classes(test_labels, cfg["task_type"])
    if inferred_n > cfg["num_classes"]:
        logger.warning(
            f"Test data contains class index {inferred_n - 1} but the model "
            f"was trained with only {cfg['num_classes']} output neurons. "
            "Predictions for unseen classes will be incorrect."
        )

    logger.info(
        f"Tokenising {len(test_texts):,} sequences "
        f"(max_length={cfg['max_length']}) …"
    )
    test_dataset = ToxicityDataset(
        test_texts, test_labels, tokenizer, cfg["max_length"]
    )

    # ── 6. Inference ──────────────────────────────────────────────────────
    logger.info("Running inference …")
    logits = _run_inference(model, test_dataset, cfg["batch_size"], device,
                            output_dir=args.output_dir)

    # ── 7. Predictions ────────────────────────────────────────────────────
    predicted_labels = np.argmax(logits, axis=-1).tolist()

    # ── 8. Classification report ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Evaluation Results")
    print(f"  Model  : {args.model_dir}")
    print(f"  Test   : {args.test_data}")
    print(f"  Task   : {cfg['task_type']}  ({cfg['num_classes']} classes)")
    print("=" * 65)

    # Trim label_names to avoid index errors if the test set has fewer classes
    present_classes = sorted(set(test_labels) | set(predicted_labels))
    safe_label_names = (
        label_names
        if len(label_names) >= len(present_classes)
        else [f"Class-{i}" for i in present_classes]
    )

    report_str = classification_report(
        test_labels,
        predicted_labels,
        target_names=safe_label_names,
        digits=4,
        zero_division=0,
    )
    print(report_str)

    # ── 9. Compute full metrics dict ──────────────────────────────────────
    from utils import compute_all_metrics
    metrics = compute_all_metrics(
        test_labels,
        predicted_labels,
        task_type=cfg["task_type"],
        label_names=safe_label_names,
    )
    # Attach evaluation context for traceability
    metrics["_meta"] = {
        "model_dir": str(args.model_dir),
        "test_data": str(args.test_data),
        "task_type": cfg["task_type"],
        "num_classes": cfg["num_classes"],
    }

    # ── 10. Save outputs ──────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # eval_metrics.json
    metrics_path = out / "eval_metrics.json"
    save_metrics(metrics, str(metrics_path))
    logger.info(f"Metrics saved → {metrics_path}")

    # confusion_matrix.png
    cm_path = out / "confusion_matrix.png"
    plot_confusion_matrix(
        test_labels,
        predicted_labels,
        label_names=safe_label_names,
        title=f"Confusion Matrix — {Path(args.model_dir).parent.name}",
        save_path=str(cm_path),
    )
    logger.info(f"Confusion matrix saved → {cm_path}")

    # predictions.csv — useful for error analysis
    preds_df = pd.DataFrame(
        {
            "text":            test_texts,
            "true_label":      test_labels,
            "predicted_label": predicted_labels,
            "correct":         [t == p for t, p in zip(test_labels, predicted_labels)],
        }
    )
    preds_path = out / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"Predictions saved → {preds_path}")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\nKey metrics (test set):")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)  : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    if cfg["task_type"] == "binary":
        print(f"  F1 (binary) : {metrics['f1_binary']:.4f}")
        print(f"  Precision   : {metrics['precision_binary']:.4f}")
        print(f"  Recall      : {metrics['recall_binary']:.4f}")
    print(f"\nFull results written to: {out}")


if __name__ == "__main__":
    main()
