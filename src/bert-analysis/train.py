"""
train.py — CLI entry point for fine-tuning a toxicity classifier with
           Stratified K-Fold cross-validation.

Training pipeline
-----------------
1. Load data:
     --dataset mode     : single file → stratified 85/15 train/test split.
     --train_data mode  : train file + val file; val file is the final test set.
                          No re-splitting of the training data is performed.
2. Run 5-fold Stratified CV on the training data only.
   Each fold trains an independent model with early stopping on the
   fold-internal validation slice.  The held-out test file is never touched.
3. Derive the average best epoch across all CV folds.
4. Train a final model on the full training data for that many epochs
   (no early stopping — no data is withheld).
5. Save the final model, tokenizer, CV metrics, and a training config.
6. The held-out test file (splits/test.parquet) is left for evaluate.py.

Input modes
-----------
Mode A — single file (--dataset):
    python train.py \\
        --dataset    processed_data/jigsaw/jigsaw.parquet \\
        --output_dir outputs/jigsaw_binary                \\
        --task_type  binary

Mode B — pre-split files (--train_data + --val_data):
    python train.py \\
        --train_data processed_data/wot/x_train.parquet      \\
        --val_data   processed_data/wot/x_validation.parquet \\
        --output_dir outputs/wot_binary                      \\
        --task_type  binary

    Here --val_data is the FINAL HELD-OUT TEST SET.
    It is never seen during cross-validation or final model training.

Artefacts written to --output_dir
----------------------------------
    model/                 fine-tuned weights + tokenizer
    splits/test.parquet    held-out test set (for evaluate.py)
    splits/train.parquet   training pool (--dataset mode only)
    training_config.json   all hyperparameters + CV settings
    cv_metrics.json        per-fold best metric and best epoch
    train_metrics.json     alias of cv_metrics.json (for compatibility)
    training_curve.png     bar chart of per-fold CV metric
"""

import argparse
import gc
import json
import sys
from pathlib import Path

# ── Windows DLL fix ──────────────────────────────────────────────────────────
import pyarrow as _pa
_pa.array([0])   # forces Arrow thread-pool + allocator init
del _pa
# ─────────────────────────────────────────────────────────────────────────────

import torch
from sklearn.model_selection import StratifiedKFold
from transformers import EarlyStoppingCallback, TrainingArguments

from data import (
    ToxicityDataset,
    apply_label_scheme,
    get_texts,
    infer_num_classes,
    light_clean,
    load_dataframe,
    make_splits,
    prepare_labels,
)
from model import WeightedTrainer, build_model, build_tokenizer, save_model
from utils import (
    compute_metrics_binary,
    get_class_weights,
    get_device,
    get_logger,
    make_compute_metrics_mc,
    save_metrics,
    set_seed,
)

logger = get_logger("train")

# Cross-validation settings — fixed for reproducibility across all runs
CV_SEED  = 7524
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define and parse all command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a transformer toxicity classifier using "
            f"{CV_FOLDS}-fold stratified cross-validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Pre-split files — val_data becomes the final held-out test set\n"
            "  python train.py --train_data data/wot_train.parquet \\\n"
            "                  --val_data   data/wot_val.parquet   \\\n"
            "                  --output_dir outputs/wot_binary     \\\n"
            "                  --task_type  binary\n\n"
            "  # Single file — 85/15 train/test split applied automatically\n"
            "  python train.py --dataset data/jigsaw.parquet \\\n"
            "                  --output_dir outputs/jigsaw_binary \\\n"
            "                  --task_type  binary\n"
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
            "A stratified 85/15 train/test split is applied automatically. "
            "The test slice is written to splits/test.parquet and never used "
            "during CV or final training. "
            "Mutually exclusive with --train_data / --val_data."
        ),
    )
    data_grp.add_argument(
        "--train_data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the training CSV or parquet. "
            "Used exclusively for cross-validation and final model training. "
            "No further splitting of this file is performed."
        ),
    )
    data_grp.add_argument(
        "--val_data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the held-out test CSV or parquet. "
            "This file is the FINAL TEST SET — it is never seen during "
            "cross-validation or final model training. "
            "Saved to splits/test.parquet for evaluate.py."
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
    data_grp.add_argument(
        "--label_scheme",
        type=str,
        default=None,
        choices=["wot3", "dota3", "jigsaw3"],
        metavar="SCHEME",
        help=(
            "Named label remapping scheme applied before training. "
            "Remaps raw dataset labels to the standardised 3-class space "
            "(0=Non-Toxic, 1=Mild, 2=Severe). "
            "Choices: wot3, dota3, jigsaw3. "
            "Omit for binary runs or when the label column is already correct."
        ),
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
        help="Number of output classes (inferred from training data if omitted).",
    )
    task_grp.add_argument(
        "--label_names",
        type=str,
        default=None,
        metavar="NAME,NAME,...",
        help=(
            "Comma-separated class names for reports/plots "
            "(e.g. 'Non-Toxic,Toxic')."
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
    train_grp.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Maximum epochs per CV fold (early stopping may end a fold sooner).",
    )
    train_grp.add_argument("--batch_size",   type=int,   default=32)
    train_grp.add_argument("--lr",           type=float, default=2e-5,
                           help="AdamW learning rate.")
    train_grp.add_argument("--weight_decay", type=float, default=0.01)
    train_grp.add_argument("--warmup_ratio", type=float, default=0.1,
                           help="Fraction of total steps used for LR warm-up.")
    train_grp.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help=(
            "Per-fold early stopping patience: stop a fold's training if the "
            "fold-internal validation metric does not improve for N epochs."
        ),
    )
    train_grp.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(args: argparse.Namespace) -> tuple:
    """
    Load the training pool and the held-out test set.

    Mode A — single file (``--dataset``):
        Applies a stratified 85/15 split.  The 85 % pool is used for CV
        and final training; the 15 % test set is saved to
        ``splits/test.parquet`` and never touched again.

    Mode B — pre-split files (``--train_data`` + ``--val_data``):
        The training file is used as-is.  The ``--val_data`` file is the
        final held-out test set — it is written to ``splits/test.parquet``
        and excluded from all training and CV steps.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    split_dir = Path(args.output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        logger.info(f"Loading single dataset file: {args.dataset}")
        df = load_dataframe(args.dataset)
        if "clean_message" not in df.columns:
            df["clean_message"] = df[args.text_col].apply(light_clean)
        if "comment_length" not in df.columns:
            df["comment_length"] = df[args.text_col].str.split().str.len()

        if args.label_scheme:
            logger.info(f"Applying label scheme '{args.label_scheme}' …")
            df = apply_label_scheme(df, args.label_scheme, args.label_col)
            logger.info(
                f"Remapped label distribution: "
                + str(df[args.label_col].value_counts().sort_index().to_dict())
            )

        logger.info("Applying stratified 85/15 train/test split …")
        train_df, test_df = make_splits(
            df,
            label_col=args.label_col,
            seed=args.seed,
        )
        train_df.to_parquet(split_dir / "train.parquet", index=False)
        test_df.to_parquet( split_dir / "test.parquet",  index=False)
        logger.info(
            f"Splits saved → {split_dir}  "
            f"(train={len(train_df):,}  test={len(test_df):,})"
        )
        return train_df, test_df

    if args.train_data and args.val_data:
        logger.info(
            f"Loading training file : {args.train_data}\n"
            f"  Held-out test file  : {args.val_data}"
        )
        train_df = load_dataframe(args.train_data)
        test_df  = load_dataframe(args.val_data)

        if args.label_scheme:
            logger.info(f"Applying label scheme '{args.label_scheme}' …")
            train_df = apply_label_scheme(train_df, args.label_scheme, args.label_col)
            test_df  = apply_label_scheme(test_df,  args.label_scheme, args.label_col)
            logger.info(
                f"Remapped train label distribution: "
                + str(train_df[args.label_col].value_counts().sort_index().to_dict())
            )

        # Save a copy so evaluate.py can find it at the standard path
        test_df.to_parquet(split_dir / "test.parquet", index=False)
        logger.info(
            f"train={len(train_df):,}  test (held-out)={len(test_df):,}"
        )
        return train_df, test_df

    logger.error(
        "Provide either --dataset (single file) or "
        "both --train_data and --val_data (pre-split files)."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _run_cv(
    train_df,
    tokenizer,
    args: argparse.Namespace,
    num_classes: int,
    class_weights,
    device,
    best_metric: str,
) -> tuple:
    """
    Run Stratified K-Fold cross-validation on *train_df*.

    Each fold:
      1. Splits *train_df* into a fold-train and fold-val slice using
         StratifiedKFold (stratified on the binary label to handle rare
         severity classes safely).
      2. Trains a fresh model with :class:`WeightedTrainer` and
         :class:`EarlyStoppingCallback` on the fold-train slice, evaluating
         after every epoch on the fold-val slice.
      3. Records the best validation metric and the epoch at which it
         occurred.
      4. Deletes the model and clears GPU memory before the next fold.

    The held-out test set is never seen inside this function.

    Parameters
    ----------
    train_df    : full training DataFrame (all CV folds drawn from here)
    tokenizer   : pre-loaded HuggingFace tokenizer (shared across folds)
    args        : parsed CLI arguments
    num_classes : number of output classes
    class_weights : 1-D tensor of per-class weights (from full training data)
    device      : torch.device
    best_metric : ``"f1"`` (binary) or ``"f1_macro"`` (multiclass)

    Returns
    -------
    tuple[list[dict], int]
        ``(fold_metrics, avg_best_epoch)``

        *fold_metrics* — one dict per fold with keys:
            ``fold``, ``best_<metric>``, ``best_epoch``
        *avg_best_epoch* — rounded mean of best epochs across folds;
            used as the fixed epoch count for final model training.
    """
    all_labels    = prepare_labels(train_df, args.label_col, args.task_type)
    # Stratify on binary labels to avoid failures with rare ordinal classes
    binary_labels = [1 if lbl > 0 else 0 for lbl in all_labels]

    kfold = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_SEED,
    )
    compute_metrics = (
        compute_metrics_binary
        if args.task_type == "binary"
        else make_compute_metrics_mc(num_classes)
    )
    metric_key  = f"eval_{best_metric}"
    fold_metrics = []
    best_epochs  = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(train_df, binary_labels), start=1
    ):
        logger.info(
            f"── Fold {fold}/{CV_FOLDS} "
            f"(train={len(train_idx):,}  val={len(val_idx):,}) ──"
        )

        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val   = train_df.iloc[val_idx  ].reset_index(drop=True)

        fold_train_labels = prepare_labels(fold_train, args.label_col, args.task_type)
        fold_val_labels   = prepare_labels(fold_val,   args.label_col, args.task_type)

        fold_train_dataset = ToxicityDataset(
            get_texts(fold_train, args.text_col),
            fold_train_labels,
            tokenizer,
            args.max_length,
        )
        fold_val_dataset = ToxicityDataset(
            get_texts(fold_val, args.text_col),
            fold_val_labels,
            tokenizer,
            args.max_length,
        )

        model    = build_model(args.model_name, num_labels=num_classes)
        fold_dir = Path(args.output_dir) / "checkpoints" / f"fold_{fold}"

        fold_training_args = TrainingArguments(
            output_dir                  = str(fold_dir),
            num_train_epochs            = args.epochs,
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size  = args.batch_size,
            learning_rate               = args.lr,
            weight_decay                = args.weight_decay,
            warmup_ratio                = args.warmup_ratio,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = best_metric,
            greater_is_better           = True,
            logging_steps               = 50,
            seed                        = args.seed,
            report_to                   = "none",
        )

        trainer = WeightedTrainer(
            class_weights   = class_weights,
            device          = device,
            model           = model,
            args            = fold_training_args,
            train_dataset   = fold_train_dataset,
            eval_dataset    = fold_val_dataset,
            compute_metrics = compute_metrics,
            callbacks       = [
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                )
            ],
        )
        trainer.train()

        # Pull the best validation result and the epoch it occurred at
        val_entries = [
            e for e in trainer.state.log_history if metric_key in e
        ]
        if val_entries:
            best_entry = max(val_entries, key=lambda e: e[metric_key])
            best_val   = best_entry[metric_key]
            # "epoch" in the log entry is a float (e.g. 3.0 at end of epoch 3)
            best_epoch = int(round(best_entry.get("epoch", len(val_entries))))
        else:
            best_val   = 0.0
            best_epoch = args.epochs

        fold_metrics.append({
            "fold":                    fold,
            f"best_{best_metric}":     best_val,
            "best_epoch":              best_epoch,
        })
        best_epochs.append(best_epoch)
        logger.info(
            f"  Fold {fold} — best val {best_metric}: {best_val:.4f}"
            f"  (epoch {best_epoch})"
        )

        # Free GPU memory before the next fold
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_best_epoch = max(1, round(sum(best_epochs) / len(best_epochs)))
    mean_val = sum(m[f"best_{best_metric}"] for m in fold_metrics) / len(fold_metrics)
    logger.info(
        f"CV complete — mean {best_metric}: {mean_val:.4f}  "
        f"avg best epoch: {avg_best_epoch}"
    )
    return fold_metrics, avg_best_epoch


# ---------------------------------------------------------------------------
# CV results plot
# ---------------------------------------------------------------------------

def _plot_cv_results(
    fold_metrics: list,
    metric_name: str,
    title: str,
    save_path: str,
) -> None:
    """Save a bar chart of per-fold validation metrics with a mean line."""
    import matplotlib.pyplot as plt

    folds  = [m["fold"] for m in fold_metrics]
    scores = [m[f"best_{metric_name}"] for m in fold_metrics]
    mean   = sum(scores) / len(scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [f"Fold {f}" for f in folds],
        scores,
        color="steelblue",
        alpha=0.8,
        edgecolor="white",
    )
    ax.axhline(
        mean, color="tomato", linestyle="--",
        linewidth=1.5, label=f"Mean = {mean:.4f}",
    )
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, min(1.05, max(scores) + 0.12))
    ax.set_ylabel(metric_name)
    ax.set_title(f"{title} — {CV_FOLDS}-Fold CV ({metric_name})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end cross-validated training pipeline.

    Steps
    -----
    1.  Parse CLI arguments and set random seed.
    2.  Load the training pool and held-out test set via :func:`_load_data`.
    3.  Prepare labels and determine ``num_classes``.
    4.  Load the tokenizer once (shared across all CV folds).
    5.  Compute class weights from the full training pool.
    6.  Run :func:`_run_cv` — 5-fold stratified CV with per-fold early
        stopping.  The held-out test set is never accessed here.
    7.  Train a final model on the full training pool for
        ``avg_best_epoch`` epochs (no early stopping, no eval).
    8.  Save the final model, tokenizer, CV metrics, and training config.
    9.  Print a reminder to run evaluate.py on the held-out test set.
    """
    args = parse_args()

    # ── 1. Reproducibility ────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}  |  Model seed: {args.seed}  |  CV seed: {CV_SEED}")

    # ── 2. Parse optional label names ─────────────────────────────────────
    label_names = None
    if args.label_names:
        label_names = [n.strip() for n in args.label_names.split(",")]

    # ── 3. Load data ──────────────────────────────────────────────────────
    train_df, test_df = _load_data(args)
    logger.info(
        f"Training pool : {len(train_df):,} rows  |  "
        f"Held-out test : {len(test_df):,} rows"
    )

    # ── 4. Prepare labels ─────────────────────────────────────────────────
    train_labels = prepare_labels(train_df, args.label_col, args.task_type)
    num_classes  = args.num_classes or infer_num_classes(train_labels, args.task_type)
    logger.info(f"Task: {args.task_type}  |  Classes: {num_classes}")

    if label_names is None:
        label_names = (
            ["Non-Toxic", "Toxic"]
            if args.task_type == "binary"
            else [f"Class-{i}" for i in range(num_classes)]
        )

    # ── 5. Tokenizer ──────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = build_tokenizer(args.model_name)

    # ── 6. Class weights (derived from full training pool only) ───────────
    class_weights = get_class_weights(train_labels, n_classes=num_classes)
    logger.info(
        "Class weights: "
        + "  ".join(
            f"{label_names[i]}={class_weights[i]:.3f}"
            for i in range(num_classes)
        )
    )

    best_metric = "f1" if args.task_type == "binary" else "f1_macro"

    # ── 7. Cross-validation ───────────────────────────────────────────────
    logger.info(
        f"Starting {CV_FOLDS}-fold stratified CV  "
        f"(seed={CV_SEED}  max_epochs_per_fold={args.epochs}  "
        f"patience={args.early_stopping_patience}) …"
    )
    fold_metrics, avg_best_epoch = _run_cv(
        train_df, tokenizer, args, num_classes, class_weights, device, best_metric
    )

    # ── 8. Final model on full training pool ──────────────────────────────
    logger.info(
        f"Training final model on full training pool "
        f"({len(train_df):,} rows, {avg_best_epoch} epoch(s)) …"
    )
    train_texts   = get_texts(train_df, args.text_col)
    train_dataset = ToxicityDataset(
        train_texts, train_labels, tokenizer, args.max_length
    )

    final_model   = build_model(args.model_name, num_labels=num_classes)
    final_ckpt    = Path(args.output_dir) / "checkpoints" / "final"
    final_tr_args = TrainingArguments(
        output_dir                  = str(final_ckpt),
        num_train_epochs            = avg_best_epoch,
        per_device_train_batch_size = args.batch_size,
        learning_rate               = args.lr,
        weight_decay                = args.weight_decay,
        warmup_ratio                = args.warmup_ratio,
        # No eval set for the final model — train straight through
        eval_strategy               = "no",
        save_strategy               = "epoch",
        logging_steps               = 50,
        seed                        = args.seed,
        report_to                   = "none",
    )

    final_trainer = WeightedTrainer(
        class_weights = class_weights,
        device        = device,
        model         = final_model,
        args          = final_tr_args,
        train_dataset = train_dataset,
    )
    final_trainer.train()
    logger.info("Final model training complete.")

    # ── 9. Save artefacts ─────────────────────────────────────────────────
    model_dir = Path(args.output_dir) / "model"
    save_model(final_trainer.model, tokenizer, str(model_dir))
    logger.info(f"Model + tokenizer saved → {model_dir}")

    # training_config.json
    config = {
        "model_name":        args.model_name,
        "task_type":         args.task_type,
        "num_classes":       num_classes,
        "label_names":       label_names,
        "max_length":        args.max_length,
        "text_col":          args.text_col,
        "label_col":         args.label_col,
        "label_scheme":      args.label_scheme,
        "seed":              args.seed,
        "batch_size":        args.batch_size,
        "epochs":            args.epochs,
        "lr":                args.lr,
        "cv_folds":          CV_FOLDS,
        "cv_seed":           CV_SEED,
        "avg_best_epoch":    avg_best_epoch,
    }
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    logger.info(f"Training config saved → {config_path}")

    # cv_metrics.json  (also written as train_metrics.json for compatibility)
    cv_payload = {
        "cv_fold_results":  fold_metrics,
        "avg_best_epoch":   avg_best_epoch,
        "config":           config,
    }
    cv_path = Path(args.output_dir) / "cv_metrics.json"
    save_metrics(cv_payload, str(cv_path))
    save_metrics(cv_payload, str(Path(args.output_dir) / "train_metrics.json"))
    logger.info(f"CV metrics saved → {cv_path}")

    # training_curve.png  — per-fold bar chart
    curve_path = Path(args.output_dir) / "training_curve.png"
    _plot_cv_results(
        fold_metrics,
        metric_name=best_metric,
        title=Path(args.output_dir).name,
        save_path=str(curve_path),
    )
    logger.info(f"CV results chart saved → {curve_path}")

    # ── 10. Summary ───────────────────────────────────────────────────────
    mean_val = sum(m[f"best_{best_metric}"] for m in fold_metrics) / len(fold_metrics)
    test_path = Path(args.output_dir) / "splits" / "test.parquet"
    logger.info(
        f"\n{'─'*60}\n"
        f"  CV mean {best_metric:<12s}: {mean_val:.4f}\n"
        f"  Final model epochs    : {avg_best_epoch}\n"
        f"{'─'*60}\n"
        f"  Held-out test set     : {test_path}\n"
        f"  Run evaluation next:\n"
        f"    python evaluate.py \\\n"
        f"        --model_dir  {model_dir} \\\n"
        f"        --test_data  {test_path} \\\n"
        f"        --output_dir {Path(args.output_dir) / 'eval'}\n"
    )


if __name__ == "__main__":
    main()
