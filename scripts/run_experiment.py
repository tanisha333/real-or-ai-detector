#!/usr/bin/env python
"""End-to-end run: train on SD 1.4 subset, evaluate on all test generators.

Usage (from project root):
    python scripts/run_experiment.py

This script is intentionally self-contained and verbose so every step is
visible in the console output. It does NOT require any CLI flags – all paths
and hyperparameters are set here for reproducibility.

Dataset layout expected:
    data/raw/train/real/      <- 16 real images
    data/raw/train/fake/      <- 16 SD 1.4 images
    data/raw/val/real/        <- 8 real images
    data/raw/val/fake/        <- 8 SD 1.4 images
    data/raw/test/real/       <- 8 real images
    data/raw/test/<gen>/      <- 24 images per unseen generator folder
                                 (adm, biggan, glide, midjourney,
                                  stablediffusionv15, vqdm, wukong)
"""

import json
import logging
import sys
import os
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout on Windows so Unicode chars in log lines don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ["MPLCONFIGDIR"] = str(ROOT / "data" / ".mpl_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

import torch
import numpy as np

from src.training.config import TrainingConfig
from src.training.trainer import train, set_seed
from src.models.detector import UniversalFakeDetector
from src.data.dataset import BinaryImageDataset
from src.utils.metrics import compute_evaluation_metrics, format_metrics_table


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Use ViT-B/32 for this quick experiment run (faster than ViT-L/14).
# The architecture is identical; ViT-L/14 would give stronger results on a
# full-scale dataset.
CFG = TrainingConfig(
    model_name="openai/clip-vit-base-patch32",
    train_dir="data/raw/train",
    val_dir="data/raw/val",
    test_dir="data/raw/test",
    feature_cache_dir="data/features",
    checkpoint_dir="checkpoints",
    results_dir="results",
    batch_size=16,
    num_workers=0,
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=1e-4,
    scheduler="cosine",
    warmup_epochs=1,
    num_epochs=30,
    early_stopping_patience=8,
    use_feature_cache=True,
    seed=42,
    threshold=0.5,
    log_every_n_steps=999,   # keep output clean
    save_best_only=True,
    eval_linear_probe=True,
    eval_nearest_neighbor=True,
)

TRAIN_GENERATOR = "stable_diffusion_v_1_4"   # the generator seen at training time
UNSEEN_GENERATORS = [                          # all test generators NOT seen at train time
    "adm", "biggan", "glide", "midjourney",
    "stablediffusionv15", "vqdm", "wukong",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 – Summarise the dataset
# ─────────────────────────────────────────────────────────────────────────────
def summarise_dataset():
    print("\n" + "=" * 70)
    print("STEP 0: DATASET SUMMARY")
    print("=" * 70)

    for split, path in [("train", CFG.train_dir),
                        ("val",   CFG.val_dir),
                        ("test",  CFG.test_dir)]:
        try:
            ds = BinaryImageDataset(root_dir=path)
            counts = ds.get_class_counts()
            generators = sorted(set(s["generator"] for s in ds.samples))
            print(f"\n  [{split.upper()}]  total={counts['total']:4d}  "
                  f"real={counts['real']:3d}  fake={counts['fake']:3d}")
            print(f"    Generators: {generators}")
        except Exception as e:
            print(f"  [{split.upper()}]  ERROR: {e}")

    print()
    print("  Training on: Stable Diffusion v1.4 images only.")
    print("  Unseen test generators: ADM, BigGAN, GLIDE, Midjourney,")
    print("    Stable Diffusion v1.5, VQDM, Wukong")
    print("  --> This tests cross-generator generalisation (paper requirement)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Train
# ─────────────────────────────────────────────────────────────────────────────
def run_training():
    print("\n" + "=" * 70)
    print("STEP 1: TRAINING LINEAR PROBE")
    print("=" * 70)
    print(f"  Backbone   : {CFG.model_name} (FROZEN)")
    print(f"  Optimizer  : {CFG.optimizer}  lr={CFG.learning_rate}")
    print(f"  Scheduler  : {CFG.scheduler}  warmup={CFG.warmup_epochs} epoch(s)")
    print(f"  Max epochs : {CFG.num_epochs}  (early stop patience={CFG.early_stopping_patience})")
    print(f"  Cache dir  : {CFG.feature_cache_dir}")

    set_seed(CFG.seed)
    detector = UniversalFakeDetector(model_name=CFG.model_name, dropout=CFG.dropout)

    history = train(CFG, detector=detector)

    print("\n  Training complete.")
    print(f"  Best val accuracy: {history['best_val_acc'] * 100:.1f}%  "
          f"(epoch {history['best_epoch']} / {len(history['train']['loss'])})")
    return detector, history


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Per-generator test evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(detector: UniversalFakeDetector):
    """Evaluates the trained linear probe on the full test set and per generator."""
    from src.models.feature_extractor import extract_and_cache_features, load_cached_features
    from src.data.dataset import create_dataloader
    from src.models.detector import NearestNeighborClassifier
    from src.models.feature_extractor import load_cached_features
    from src.training.trainer import prepare_feature_cache
    import torch.nn.functional as F

    print("\n" + "=" * 70)
    print("STEP 2: EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)
    detector.eval()

    model_tag = CFG.model_name.replace("/", "_").replace("-", "_")
    cache_dir = Path(CFG.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    test_cache_path = str(cache_dir / f"test_{model_tag}.pt")

    # Extract or load test features
    if Path(test_cache_path).exists():
        logger.info("Loading cached test features from %s", test_cache_path)
        test_cache = load_cached_features(test_cache_path)
    else:
        logger.info("Extracting test features from %s ...", CFG.test_dir)
        test_ds = BinaryImageDataset(root_dir=CFG.test_dir)
        logger.info("  Test set: %s", test_ds.get_class_counts())
        test_dl = create_dataloader(test_ds, batch_size=CFG.batch_size, shuffle=False)
        test_cache = extract_and_cache_features(detector, test_dl, test_cache_path, device)

    test_feats  = test_cache["features"]
    test_labels = test_cache["labels"].long()
    test_gens   = test_cache.get("generators", ["unknown"] * len(test_labels))

    # ── Linear probe predictions ──────────────────────────────────────────────
    detector.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(test_feats), 128):
            batch = test_feats[start:start + 128].to(device)
            logits = detector.forward_features(batch)
            probs  = torch.sigmoid(logits).squeeze(1).cpu()
            all_probs.append(probs)
    probs_arr = torch.cat(all_probs).numpy()
    preds_arr = (probs_arr >= CFG.threshold).astype(int)
    y_true    = test_labels.numpy()

    # ── Nearest neighbour predictions ─────────────────────────────────────────
    train_cache_path = str(cache_dir / f"train_{model_tag}.pt")
    train_cache = load_cached_features(train_cache_path)
    nn_clf = NearestNeighborClassifier()
    nn_clf.fit(train_cache["features"], train_cache["labels"])

    nn_preds_list = []
    test_feats_norm = F.normalize(test_feats, p=2, dim=-1)
    for start in range(0, len(test_feats_norm), 256):
        batch = test_feats_norm[start:start + 256]
        nn_preds_list.append(nn_clf.predict(batch).numpy())
    nn_preds = np.concatenate(nn_preds_list)

    # ── Collect results ───────────────────────────────────────────────────────
    unique_gens = sorted(set(test_gens))
    results = {}

    for classifier_name, preds, probs in [
        ("linear_probe",      preds_arr, probs_arr),
        ("nearest_neighbor",  nn_preds,  None),
    ]:
        overall = compute_evaluation_metrics(y_true, preds, probs)
        per_gen = {}
        for gen in unique_gens:
            mask = np.array([g == gen for g in test_gens])
            if mask.sum() == 0:
                continue
            per_gen[gen] = compute_evaluation_metrics(
                y_true[mask], preds[mask],
                probs[mask] if probs is not None else None,
            )
        results[classifier_name] = {"overall": overall, "per_generator": per_gen}

    # ── Save JSON report ──────────────────────────────────────────────────────
    results_dir = Path(CFG.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [_convert(v) for v in obj]
        return obj

    report_path = results_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(_convert(results), f, indent=2)

    # ── Save plots ────────────────────────────────────────────────────────────
    _save_plots(y_true, preds_arr, probs_arr, nn_preds, results_dir)

    return results, y_true, test_gens


def _save_plots(y_true, lp_preds, lp_probs, nn_preds, results_dir):
    """Saves confusion matrix and ROC plots without crashing on display-less systems."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        from src.utils.visualization import plot_confusion_matrix, plot_roc_curve
        from src.utils.metrics import compute_evaluation_metrics

        for name, preds, probs in [("linear_probe", lp_preds, lp_probs),
                                    ("nearest_neighbor", nn_preds, None)]:
            m = compute_evaluation_metrics(y_true, preds, probs)
            plot_confusion_matrix(
                m["confusion_matrix"]["matrix"],
                save_path=str(results_dir / f"{name}_confusion_matrix.png"),
                title=f"Confusion Matrix — {name.replace('_', ' ').title()}",
            )
            if probs is not None and len(np.unique(y_true)) > 1:
                plot_roc_curve(y_true, probs,
                               save_path=str(results_dir / f"{name}_roc_curve.png"))
    except Exception as e:
        logger.warning("Could not save plots: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Print human-readable results report
# ─────────────────────────────────────────────────────────────────────────────
def print_results(results, y_true, test_gens):
    print("\n" + "=" * 70)
    print("STEP 3: RESULTS")
    print("=" * 70)

    for clf_name, clf_results in results.items():
        print(f"\n{'─'*70}")
        print(f"  CLASSIFIER: {clf_name.replace('_', ' ').upper()}")
        print(f"{'─'*70}")

        overall = clf_results["overall"]
        per_gen = clf_results.get("per_generator", {})

        # ── Overall metrics ───────────────────────────────────────────────────
        auc_str = (f"{overall['roc_auc'] * 100:.1f}%"
                   if overall.get("roc_auc") is not None else "N/A (no probs)")
        ap_str  = (f"{overall['average_precision'] * 100:.1f}%"
                   if overall.get("average_precision") is not None else "N/A")

        print(f"\n  OVERALL TEST METRICS  (n={overall['sample_counts']['total']})")
        print(f"    Accuracy         : {overall['accuracy']*100:.1f}%")
        print(f"    Precision        : {overall['precision']*100:.1f}%")
        print(f"    Recall           : {overall['recall']*100:.1f}%")
        print(f"    Specificity      : {overall['specificity']*100:.1f}%")
        print(f"    F1-Score         : {overall['f1_score']*100:.1f}%")
        print(f"    ROC-AUC          : {auc_str}")
        print(f"    Average Precision: {ap_str}")
        cm = overall["confusion_matrix"]
        print(f"\n  Confusion Matrix:")
        print(f"    True  Negatives (real  -> real ): {cm['tn']:4d}")
        print(f"    False Positives (real  -> fake ): {cm['fp']:4d}")
        print(f"    False Negatives (fake  -> real ): {cm['fn']:4d}")
        print(f"    True  Positives (fake  -> fake ): {cm['tp']:4d}")

        # ── Per-generator breakdown ───────────────────────────────────────────
        if per_gen:
            print(f"\n  PER-GENERATOR ACCURACY:")
            print(f"    {'Generator':22s}  {'N':>4s}  {'Acc':>7s}  {'F1':>7s}  "
                  f"{'Seen at train?':>14s}")
            print(f"    {'─'*22}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*14}")

            for gen, m in sorted(per_gen.items()):
                n    = m["sample_counts"]["total"]
                acc  = m["accuracy"] * 100
                f1   = m["f1_score"] * 100
                seen = "YES (SD 1.4)" if gen in ("real", "ai_generator") else "NO (unseen)"
                # "real" label means the real images bucket, not a fake generator
                if gen == "real":
                    print(f"    {'real (natural)':22s}  {n:4d}  {acc:6.1f}%  {f1:6.1f}%  {'—':>14s}")
                else:
                    print(f"    {gen:22s}  {n:4d}  {acc:6.1f}%  {f1:6.1f}%  {seen:>14s}")

    # ── Unseen-generator check ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  UNSEEN GENERATOR EVALUATION (Paper Requirement Check)")
    print(f"{'─'*70}")
    test_gens_set = set(test_gens) - {"real"}
    print(f"  Generator(s) seen at training   : stable_diffusion_v_1_4 (SD 1.4)")
    print(f"  Generator(s) present in test set: {sorted(test_gens_set)}")
    unseen_present = test_gens_set - {"ai_generator"}  # ai_generator = SD 1.4 flat-folder fallback
    if unseen_present:
        print(f"\n  [OK] The test set includes {len(unseen_present)} unseen generators.")
        print(f"       The paper's cross-generator generalisation requirement is MET.")
    else:
        print(f"\n  [WARN] No distinctly named unseen generators detected.")
        print(f"         Add per-generator test sub-folders to satisfy the paper requirement.")

    print(f"\n  Results saved to: {CFG.results_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Metric explanations
# ─────────────────────────────────────────────────────────────────────────────
def print_metric_explanations():
    print("\n" + "=" * 70)
    print("METRIC EXPLANATIONS (plain English)")
    print("=" * 70)
    explanations = [
        ("Accuracy",
         "What fraction of ALL images (real + fake) were labelled correctly.\n"
         "     90% = 9 out of 10 images classified right."),
        ("Precision",
         "Of all images the model flagged as AI-generated, how many actually were?\n"
         "     High precision = few false alarms on real images."),
        ("Recall (Sensitivity)",
         "Of all the actual AI-generated images in the test set, how many did\n"
         "     the model catch? High recall = few AI images slipping through undetected."),
        ("Specificity",
         "Of all the actual REAL images, how many did the model correctly\n"
         "     leave alone? Low specificity = many real images wrongly flagged."),
        ("F1-Score",
         "The harmonic mean of Precision and Recall. A single number that\n"
         "     balances both concerns. Useful when classes are imbalanced."),
        ("ROC-AUC",
         "Area Under the Receiver Operating Characteristic Curve.\n"
         "     Measures ranking quality: 1.0 = perfect, 0.5 = coin flip.\n"
         "     Crucially, it does NOT depend on the decision threshold (0.5)."),
        ("Average Precision",
         "Area under the Precision-Recall curve. Particularly informative\n"
         "     when fake images are rare (class imbalance)."),
    ]
    for name, desc in explanations:
        print(f"\n  {name}:")
        print(f"     {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  Real-or-AI Detector — End-to-End Experiment Run")
    print("  Model : " + CFG.model_name)
    print("  Device: " + str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    print("#" * 70)

    summarise_dataset()

    detector, history = run_training()

    results, y_true, test_gens = run_evaluation(detector)

    print_results(results, y_true, test_gens)

    print_metric_explanations()

    print("\n" + "#" * 70)
    print("  Experiment complete. Review results above.")
    print("  Approve to proceed to Stage 4.")
    print("#" * 70 + "\n")
