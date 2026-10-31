#!/usr/bin/env python
"""CLI entry-point: train the Universal Fake Image Detector linear probe.

Usage
-----
From the project root:

    python scripts/train.py [OPTIONS]

Options map directly to TrainingConfig fields.  All have sensible defaults.

Examples
--------
# Quick smoke-test on tiny dataset, ViT-B/32:
    python scripts/train.py --model_name openai/clip-vit-base-patch32 --num_epochs 5

# Full ViT-L/14 run matching paper settings:
    python scripts/train.py --num_epochs 20 --learning_rate 1e-3

# Resume from a specific checkpoint directory:
    python scripts/train.py --checkpoint_dir checkpoints/run2

# Disable early stopping:
    python scripts/train.py --early_stopping_patience -1
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.config import TrainingConfig
from src.training.trainer import train


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> TrainingConfig:
    """Parses CLI arguments and returns a populated TrainingConfig."""
    defaults = TrainingConfig()
    p = argparse.ArgumentParser(
        description="Train the linear probe of the Universal Fake Image Detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_name", default=defaults.model_name,
                   help="Hugging Face CLIP variant (e.g. openai/clip-vit-large-patch14).")
    p.add_argument("--dropout", type=float, default=defaults.dropout,
                   help="Dropout rate before the linear probe (0.0 = disabled).")

    # Data
    p.add_argument("--train_dir", default=defaults.train_dir,
                   help="Training image root (must contain real/ and fake/ sub-dirs).")
    p.add_argument("--val_dir", default=defaults.val_dir,
                   help="Validation image root.")
    p.add_argument("--test_dir", default=defaults.test_dir,
                   help="Test image root.")
    p.add_argument("--feature_cache_dir", default=defaults.feature_cache_dir,
                   help="Directory for pre-extracted CLIP feature caches (.pt files).")
    p.add_argument("--checkpoint_dir", default=defaults.checkpoint_dir,
                   help="Directory for saved probe checkpoints.")
    p.add_argument("--results_dir", default=defaults.results_dir,
                   help="Directory for evaluation reports and plots.")
    p.add_argument("--batch_size", type=int, default=defaults.batch_size)
    p.add_argument("--num_workers", type=int, default=defaults.num_workers)

    # Optimiser
    p.add_argument("--optimizer", default=defaults.optimizer, choices=["adamw", "sgd"])
    p.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    p.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    p.add_argument("--momentum", type=float, default=defaults.momentum)

    # Scheduler
    p.add_argument("--scheduler", default=defaults.scheduler, choices=["cosine", "step", "none"])
    p.add_argument("--warmup_epochs", type=int, default=defaults.warmup_epochs)
    p.add_argument("--step_size", type=int, default=defaults.step_size)
    p.add_argument("--step_gamma", type=float, default=defaults.step_gamma)

    # Training loop
    p.add_argument("--num_epochs", type=int, default=defaults.num_epochs)
    p.add_argument("--early_stopping_patience", type=int, default=defaults.early_stopping_patience,
                   help="Patience for early stopping (-1 to disable).")
    p.add_argument("--no_feature_cache", action="store_true",
                   help="Disable feature caching (slower; avoids disk I/O).")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--threshold", type=float, default=defaults.threshold,
                   help="Sigmoid decision threshold for binary classification.")
    p.add_argument("--log_every_n_steps", type=int, default=defaults.log_every_n_steps)
    p.add_argument("--save_all_checkpoints", action="store_true",
                   help="Save a checkpoint every epoch (default: save best only).")

    # Evaluation after training
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip final test-set evaluation after training.")
    p.add_argument("--eval_linear_probe", action="store_true", default=True)
    p.add_argument("--eval_nearest_neighbor", action="store_true", default=True)

    args = p.parse_args()

    cfg = TrainingConfig(
        model_name=args.model_name,
        dropout=args.dropout,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        feature_cache_dir=args.feature_cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        step_size=args.step_size,
        step_gamma=args.step_gamma,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        use_feature_cache=not args.no_feature_cache,
        seed=args.seed,
        threshold=args.threshold,
        log_every_n_steps=args.log_every_n_steps,
        save_best_only=not args.save_all_checkpoints,
        eval_linear_probe=not args.skip_eval and args.eval_linear_probe,
        eval_nearest_neighbor=not args.skip_eval and args.eval_nearest_neighbor,
    )
    return cfg, args.skip_eval


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg, skip_eval = parse_args()

    logging.info("=" * 60)
    logging.info("Real-or-AI Detector — Stage 3: Training")
    logging.info("=" * 60)
    logging.info("Config:")
    for field_name, field_val in cfg.__dict__.items():
        logging.info("  %-30s %s", field_name, field_val)

    history = train(cfg)

    # ── Pretty print training summary ─────────────────────────────────────────
    best_epoch = history["best_epoch"]
    best_val_acc = history["best_val_acc"] * 100
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val accuracy : {best_val_acc:.2f}%  (epoch {best_epoch})")
    print(f"  Checkpoint saved  : {cfg.checkpoint_dir}/best_probe.pt")
    print(f"{'='*60}")

    if not skip_eval:
        from src.training.evaluator import evaluate
        logging.info("\nRunning final test-set evaluation …")
        evaluate(cfg)


if __name__ == "__main__":
    main()
