"""
train.py — CLI entry point for fine-tuning a BERT toxicity classifier.

The script handles two input modes:

1. Pre-split files (recommended — avoids any risk of leakage):
       python train.py \\
           --train_data processed_data/wot/wot_train.parquet \\
           --val_data   processed_data/wot/wot_val.parquet   \\
           --output_dir outputs/wot_binary                   \\
           --task_type  binary

2. Single file (auto 70/15/15 split performed here):
       python train.py \\
           --dataset    raw_data/wot/wot.parquet \\
           --output_dir outputs/wot_binary       \\
           --task_type  binary                   \\
           --epochs     3

After training the following artefacts are written to --output_dir:
    model/           — fine-tuned weights + tokenizer (HuggingFace format)
    training_config.json  — all args needed to reproduce / evaluate later
    train_metrics.json    — per-epoch val metrics from the Trainer log
    training_curve.png    — loss + val-metric plot
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import EarlyStoppingCallback, TrainingArguments

# Local modules — all in the same directory
from data import (
    ToxicityDataset,
    get_texts,
    infer_num_classes,
    light_clean,
    load_dataframe,
    make_three_way_split,
    prepare_labels,
)
from model import WeightedTrainer, build_model, build_tokenizer, save_model
from utils import (
    compute_metrics_binary,
    get_class_weights,
    get_device,
    get_logger,
    make_compute_metrics_mc,
    plot_training_history,
    save_metrics,
    set_seed,
)

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Define and parse all command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for toxicity detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Pre-split parquets (no leakage)\n"
            "  python train.py --train_data data/wot_train.parquet \\\n"
            "                  --val_data   data/wot_val.parquet   \\\n"
            "                  --output_dir outputs/wot_binary     \\\n"
            "                  --task_type  binary\n\n"
            "  # Single CSV (auto-split)\n"
            "  python train.py --dataset data.csv --task_type binary --epochs 3\n"
        ),
    )

    # ── Data input ────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("data input")
    data_grp.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a single CSV or parquet file containing all examples. "
            "A stratified 70/15/15 train/val/test split is applied automatically. "
            "Mutually exclusive with --train_data / --val_data."
        ),
    )
    data_grp.add_argument(
        "--train_data",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the pre-split training CSV or parquet.",
    )
    data_grp.add_argument(
        "--val_data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the pre-split validation CSV or parquet. "
            "Used ONLY for early stopping — never for reported metrics."
        ),
    )
    data_grp.add_argument(
        "--text_col",
        type=str,
        default="message",
        help="Name of the raw-text column.",
    )
    data_grp.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the integer label column.",
    )

    # ── Task ──────────────────────────────────────────────────────────────
    task_grp = parser.add_argument_group("task")
    task_grp.add_argument(
        "--task_type",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help=(
            "'binary': collapse all labels > 0 to 1 (non-toxic vs. toxic). "
            "'multiclass': use the full ordinal label set."
        ),
    )
    task_grp.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help=(
            "Number of output classes. "
            "Inferred from the training data when not specified."
        ),
    )
    task_grp.add_argument(
        "--label_names",
        type=str,
        default=None,
        metavar="NAME,NAME,...",
        help=(
            "Comma-separated list of class names for reports/plots "
            "(e.g. 'Non-Toxic,Toxic'). "
            "Defaults to ['Non-Toxic','Toxic'] for binary and "
            "['Class-0',...] for multiclass."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model identifier or path to a local checkpoint.",
    )
    model_grp.add_argument(
        "--max_length",
        type=int,
        default=64,
        help=(
            "Maximum tokeniser sequence length. "
            "64 is sufficient for short game-chat (WOT/DOTA). "
            "Use 256 for long social-media text (Jigsaw)."
        ),
    )

    # ── Training hyperparameters ──────────────────────────────────────────
    train_grp = parser.add_argument_group("training")
    train_grp.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Root directory for saved model, tokenizer, and metrics.",
    )
    train_grp.add_argument("--epochs",        type=int,   default=4)
    train_grp.add_argument("--batch_size",    type=int,   default=32)
    train_grp.add_argument("--lr",            type=float, default=2e-5,
                           help="AdamW learning rate.")
    train_grp.add_argument("--weight_decay",  type=float, default=0.01)
    train_grp.add_argument("--warmup_ratio",  type=float, default=0.1,
                           help="Fraction of total steps used for LR warm-up.")
    train_grp.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Stop training if the monitored val metric does not improve for N epochs.",
    )
    train_grp.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_splits(args: argparse.Namespace) -> tuple:
    """
    Load training and validation DataFrames according to the CLI args.

    Mode A — pre-split files (``--train_data`` + ``--val_data``):
        Files are loaded directly; no additional splitting is performed.

    Mode B — single file (``--dataset``):
        A stratified 70/15/15 split is applied.  The test slice is saved
        alongside the train/val slices inside ``args.output_dir`` so that
        ``evaluate.py`` can use it later without any extra work.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df)
    """
    if args.train_data and args.val_data:
        logger.info("Loading pre-split train/val files.")
        train_df = load_dataframe(args.train_data)
        val_df   = load_dataframe(args.val_data)
        return train_df, val_df

    if args.dataset:
        logger.info(f"Loading single dataset file: {args.dataset}")
        df = load_dataframe(args.dataset)
        # Add clean_message if the column does not already exist
        if "clean_message" not in df.columns:
            df["clean_message"] = df[args.text_col].apply(light_clean)
        if "comment_length" not in df.columns:
            df["comment_length"] = df[args.text_col].str.split().str.len()

        logger.info("Creating stratified 70/15/15 train/val/test splits…")
        train_df, val_df, test_df = make_three_way_split(
            df,
            label_col=args.label_col,
            seed=args.seed,
        )

        # Persist the splits so evaluate.py can locate the test set
        split_dir = Path(args.output_dir) / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(split_dir / "train.parquet", index=False)
        val_df.to_parquet(  split_dir / "val.parquet",   index=False)
        test_df.to_parquet( split_dir / "test.parquet",  index=False)
        logger.info(
            f"Splits saved to {split_dir}  "
            f"(train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,})"
        )
        return train_df, val_df

    parser_err = (
        "Provide either --dataset (single file) or "
        "both --train_data and --val_data (pre-split files)."
    )
    logger.error(parser_err)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end training pipeline.

    Steps
    -----
    1.  Parse CLI arguments.
    2.  Set random seeds for reproducibility.
    3.  Load train / val DataFrames.
    4.  Prepare labels (binary or multi-class).
    5.  Infer or validate ``num_classes``.
    6.  Build the tokenizer and tokenise all splits into ToxicityDatasets.
    7.  Compute balanced class weights from the training labels.
    8.  Initialise ``BertForSequenceClassification`` with the correct head size.
    9.  Configure ``TrainingArguments`` — val set used for early stopping only.
    10. Train with ``WeightedTrainer``.
    11. Plot and save the training curve.
    12. Persist the model, tokenizer, training config, and val-set metrics.
    """
    args = parse_args()

    # ── 1. Reproducibility ────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}  |  Seed: {args.seed}")

    # ── 2. Parse optional label names ─────────────────────────────────────
    label_names = None
    if args.label_names:
        label_names = [n.strip() for n in args.label_names.split(",")]

    # ── 3. Load data ──────────────────────────────────────────────────────
    train_df, val_df = _load_splits(args)
    logger.info(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")

    # ── 4. Prepare labels ─────────────────────────────────────────────────
    train_labels = prepare_labels(train_df, args.label_col, args.task_type)
    val_labels   = prepare_labels(val_df,   args.label_col, args.task_type)

    # ── 5. Determine number of output classes ─────────────────────────────
    num_classes = args.num_classes or infer_num_classes(train_labels, args.task_type)
    logger.info(f"Task: {args.task_type}  |  Classes: {num_classes}")

    # Apply default label names now that we know num_classes
    if label_names is None:
        if args.task_type == "binary":
            label_names = ["Non-Toxic", "Toxic"]
        else:
            label_names = [f"Class-{i}" for i in range(num_classes)]

    # ── 6. Tokenise ───────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = build_tokenizer(args.model_name)

    train_texts = get_texts(train_df, args.text_col)
    val_texts   = get_texts(val_df,   args.text_col)

    logger.info(f"Tokenising {len(train_texts):,} train + {len(val_texts):,} val sequences …")
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset   = ToxicityDataset(val_texts,   val_labels,   tokenizer, args.max_length)

    # ── 7. Class weights ──────────────────────────────────────────────────
    # Derived from the training labels ONLY — never from val or test.
    class_weights = get_class_weights(train_labels, n_classes=num_classes)
    logger.info(
        "Class weights: " + "  ".join(
            f"{label_names[i]}={class_weights[i]:.3f}"
            for i in range(num_classes)
        )
    )

    # ── 8. Model ──────────────────────────────────────────────────────────
    logger.info(f"Building model: {args.model_name}  ({num_classes} labels)")
    model = build_model(args.model_name, num_labels=num_classes)

    # ── 9. Training arguments ─────────────────────────────────────────────
    # The metric used for early stopping and best-checkpoint selection.
    # 'f1' for binary; 'f1_macro' for multi-class (equal weight across classes).
    best_metric = "f1" if args.task_type == "binary" else "f1_macro"

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    training_args = TrainingArguments(
        output_dir                  = str(ckpt_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate               = args.lr,
        weight_decay                = args.weight_decay,
        warmup_ratio                = args.warmup_ratio,
        # Evaluate after every epoch so early stopping has a signal
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        # Load the epoch with the best val metric when training finishes
        load_best_model_at_end      = True,
        metric_for_best_model       = best_metric,
        greater_is_better           = True,
        logging_steps               = 50,
        seed                        = args.seed,
        # Disable W&B / MLflow / etc. to keep the run self-contained
        report_to                   = "none",
    )

    # ── 10. Trainer ───────────────────────────────────────────────────────
    compute_metrics = (
        compute_metrics_binary
        if args.task_type == "binary"
        else make_compute_metrics_mc(num_classes)
    )

    trainer = WeightedTrainer(
        class_weights   = class_weights,
        device          = device,
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,       # val set — early stopping only
        compute_metrics = compute_metrics,
        callbacks       = [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    logger.info("Starting training…")
    trainer.train()
    logger.info("Training complete. Best checkpoint restored.")

    # ── 11. Training curve ────────────────────────────────────────────────
    curve_path = Path(args.output_dir) / "training_curve.png"
    metric_key = f"eval_{best_metric}"
    plot_training_history(
        trainer.state.log_history,
        title=Path(args.output_dir).name,
        save_path=str(curve_path),
        metric_key=metric_key,
    )
    logger.info(f"Training curve saved → {curve_path}")

    # ── 12. Save artefacts ────────────────────────────────────────────────
    model_dir = Path(args.output_dir) / "model"
    save_model(trainer.model, tokenizer, str(model_dir))
    logger.info(f"Model + tokenizer saved → {model_dir}")

    # training_config.json — everything evaluate.py needs to reload the model
    config = {
        "model_name":   args.model_name,
        "task_type":    args.task_type,
        "num_classes":  num_classes,
        "label_names":  label_names,
        "max_length":   args.max_length,
        "text_col":     args.text_col,
        "label_col":    args.label_col,
        "seed":         args.seed,
        "batch_size":   args.batch_size,
        "epochs":       args.epochs,
        "lr":           args.lr,
    }
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    logger.info(f"Training config saved → {config_path}")

    # train_metrics.json — val-set metrics extracted from the Trainer log
    # (for tracking purposes; these are validation scores, not test scores)
    val_metric_entries = [
        e for e in trainer.state.log_history
        if metric_key in e
    ]
    train_metrics_path = Path(args.output_dir) / "train_metrics.json"
    save_metrics(
        {"val_metrics_per_epoch": val_metric_entries, "config": config},
        str(train_metrics_path),
    )
    logger.info(f"Training metrics saved → {train_metrics_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    if val_metric_entries:
        best_val = max(e[metric_key] for e in val_metric_entries)
        logger.info(f"Best val {best_metric}: {best_val:.4f}")

    logger.info(
        "\nTraining complete. Next step:\n"
        f"  python evaluate.py \\\n"
        f"      --model_dir  {model_dir} \\\n"
        f"      --test_data  <path/to/test.parquet> \\\n"
        f"      --output_dir {args.output_dir}"
    )


if __name__ == "__main__":
    main()
