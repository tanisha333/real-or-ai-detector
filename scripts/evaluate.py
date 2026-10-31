#!/usr/bin/env python
"""CLI entry-point: evaluate a trained Universal Fake Image Detector probe.

Usage
-----
From the project root:

    python scripts/evaluate.py [OPTIONS]

Examples
--------
# Evaluate the default best_probe.pt on the test set:
    python scripts/evaluate.py

# Point to a specific probe checkpoint:
    python scripts/evaluate.py --probe_checkpoint checkpoints/run2/best_probe.pt

# Evaluate only the linear probe (skip NN):
    python scripts/evaluate.py --skip_nn

# Re-extract test features even if a cache exists:
    python scripts/evaluate.py --force_recompute

# Use a lower threshold to catch more fakes (higher recall):
    python scripts/evaluate.py --threshold 0.3
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.config import TrainingConfig
from src.training.evaluator import evaluate


def parse_args():
    defaults = TrainingConfig()
    p = argparse.ArgumentParser(
        description="Evaluate a trained Real-or-AI Detector linear probe and/or NN classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model_name", default=defaults.model_name)
    p.add_argument("--train_dir", default=defaults.train_dir)
    p.add_argument("--val_dir", default=defaults.val_dir)
    p.add_argument("--test_dir", default=defaults.test_dir)
    p.add_argument("--feature_cache_dir", default=defaults.feature_cache_dir)
    p.add_argument("--checkpoint_dir", default=defaults.checkpoint_dir)
    p.add_argument("--results_dir", default=defaults.results_dir)
    p.add_argument("--probe_checkpoint", default=None,
                   help="Path to probe .pt file. Defaults to <checkpoint_dir>/best_probe.pt.")
    p.add_argument("--batch_size", type=int, default=defaults.batch_size)
    p.add_argument("--num_workers", type=int, default=defaults.num_workers)
    p.add_argument("--threshold", type=float, default=defaults.threshold)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--skip_nn", action="store_true", help="Skip nearest-neighbour evaluation.")
    p.add_argument("--skip_lp", action="store_true", help="Skip linear probe evaluation.")
    p.add_argument("--force_recompute", action="store_true",
                   help="Re-extract features even if caches exist.")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    cfg = TrainingConfig(
        model_name=args.model_name,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        feature_cache_dir=args.feature_cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        seed=args.seed,
        eval_linear_probe=not args.skip_lp,
        eval_nearest_neighbor=not args.skip_nn,
    )

    logging.info("=" * 60)
    logging.info("Real-or-AI Detector — Stage 3: Evaluation")
    logging.info("=" * 60)

    results = evaluate(
        cfg=cfg,
        probe_checkpoint=args.probe_checkpoint,
        force_recompute_features=args.force_recompute,
    )

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for classifier_name, classifier_results in results.items():
        overall = classifier_results.get("overall", {})
        acc = overall.get("accuracy", float("nan")) * 100
        auc = overall.get("roc_auc")
        auc_str = f"{auc * 100:.2f}%" if auc is not None else "N/A"
        f1 = overall.get("f1_score", float("nan")) * 100
        print(f"\n  [{classifier_name.replace('_', ' ').upper()}]")
        print(f"    Accuracy  : {acc:.2f}%")
        print(f"    ROC-AUC   : {auc_str}")
        print(f"    F1-Score  : {f1:.2f}%")

        per_gen = classifier_results.get("per_generator", {})
        if per_gen:
            print("    Per-generator accuracy:")
            for gen, m in sorted(per_gen.items()):
                print(f"      {gen:20s}  {m.get('accuracy', 0) * 100:.2f}%")

    print(f"\n  Results saved to: {cfg.results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
