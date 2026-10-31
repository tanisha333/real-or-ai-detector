"""Evaluation engine for the Universal Fake Image Detector.

Runs comprehensive evaluation of:
  1. The trained linear probe (parametric classifier).
  2. The non-parametric Nearest Neighbour (NN) classifier.

For both classifiers the evaluation is performed:
  a. Overall on the full test set.
  b. Per generator source (e.g. ADM, BigGAN, SD 1.4, …) to assess
     generalisation across unseen synthesis methods.

Results are written as JSON reports and diagnostic plots (ROC curve,
confusion matrix) to ``cfg.results_dir``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.detector import UniversalFakeDetector, NearestNeighborClassifier
from src.models.feature_extractor import extract_and_cache_features, load_cached_features
from src.data.dataset import BinaryImageDataset, create_dataloader
from src.training.config import TrainingConfig
from src.training.trainer import prepare_feature_cache, set_seed
from src.utils.metrics import compute_evaluation_metrics, format_metrics_table
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Test-set feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_test_features(
    detector: UniversalFakeDetector,
    cfg: TrainingConfig,
    device: torch.device,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Extracts or loads cached CLIP features for the held-out test set.

    The test directory may contain multiple per-generator sub-folders under
    ``fake/`` (e.g. ``test/biggan/``, ``test/adm/``) or a single flat
    ``test/real/`` + ``test/fake/`` layout.  Both layouts are supported via
    :class:`BinaryImageDataset`.

    Returns:
        Tuple of (features, labels, generators, paths) as tensors / lists.
    """
    cache_dir = Path(cfg.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_tag = cfg.model_name.replace("/", "_").replace("-", "_")
    cache_path = str(cache_dir / f"test_{model_tag}.pt")

    if not force_recompute and Path(cache_path).exists():
        logger.info("Loading cached test features from %s", cache_path)
        cache = load_cached_features(cache_path)
    else:
        logger.info("Extracting test features from %s …", cfg.test_dir)
        dataset = BinaryImageDataset(root_dir=cfg.test_dir)
        if len(dataset) == 0:
            raise RuntimeError(
                f"No images found in '{cfg.test_dir}'. "
                "Ensure the directory has 'real/' and 'fake/' sub-folders "
                "(or per-generator sub-folders under 'fake/')."
            )
        counts = dataset.get_class_counts()
        logger.info("  Test split: %d real, %d fake, %d total",
                    counts["real"], counts["fake"], counts["total"])

        dl = create_dataloader(
            dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        cache = extract_and_cache_features(detector, dl, cache_path, device)

    features = cache["features"]
    labels = cache["labels"]
    generators: List[str] = cache.get("generators", ["unknown"] * len(labels))
    paths: List[str] = cache.get("paths", [""] * len(labels))

    return features, labels, generators, paths


# ─────────────────────────────────────────────────────────────────────────────
# Per-subset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    subset_name: str,
) -> Dict[str, Any]:
    """Computes and logs metrics for a single evaluation subset."""
    metrics = compute_evaluation_metrics(y_true, y_pred, y_prob)
    table = format_metrics_table(metrics, title=f"Results — {subset_name}")
    logger.info("\n%s", table)
    return metrics


def _predict_linear_probe(
    detector: UniversalFakeDetector,
    features: torch.Tensor,
    cfg: TrainingConfig,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs batched inference with the linear probe on pre-extracted features.

    Returns:
        (predictions, probabilities) as numpy arrays.
    """
    detector.eval()
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            logits = detector.forward_features(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.concatenate(all_probs)
    preds_arr = (probs_arr >= cfg.threshold).astype(int)
    return preds_arr, probs_arr


# ─────────────────────────────────────────────────────────────────────────────
# Public evaluate function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    cfg: TrainingConfig,
    detector: Optional[UniversalFakeDetector] = None,
    device: Optional[torch.device] = None,
    probe_checkpoint: Optional[str] = None,
    force_recompute_features: bool = False,
) -> Dict[str, Any]:
    """Full Stage 3 evaluation of both classifier heads.

    Args:
        cfg:                      Training/evaluation configuration.
        detector:                 Pre-built detector (built from cfg if None).
        device:                   Compute device (auto-detected if None).
        probe_checkpoint:         Path to a saved probe .pt file to load.
                                  Defaults to ``cfg.checkpoint_dir/best_probe.pt``.
        force_recompute_features: Re-extract features even if cache exists.

    Returns:
        results: Nested dict with keys ``'linear_probe'`` and/or
                 ``'nearest_neighbor'``, each containing ``'overall'`` and
                 ``'per_generator'`` sub-dicts of evaluation metric dicts.
    """
    set_seed(cfg.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)

    # ── Build / configure detector ───────────────────────────────────────────
    if detector is None:
        detector = UniversalFakeDetector(model_name=cfg.model_name, dropout=cfg.dropout)

    # Load probe weights
    if probe_checkpoint is None:
        probe_checkpoint = str(Path(cfg.checkpoint_dir) / "best_probe.pt")

    if Path(probe_checkpoint).exists():
        logger.info("Loading linear probe from %s", probe_checkpoint)
        detector.load_probe(probe_checkpoint, map_location=str(device))
    else:
        logger.warning(
            "No probe checkpoint found at '%s'. "
            "Evaluating with randomly initialised probe weights.",
            probe_checkpoint,
        )

    detector.to(device)
    detector.eval()

    # ── Load train features (needed for NN classifier) ───────────────────────
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Test features ────────────────────────────────────────────────────────
    test_feats, test_labels, test_generators, _ = _load_test_features(
        detector, cfg, device, force_recompute=force_recompute_features
    )
    y_true = test_labels.long().numpy()
    unique_generators = sorted(set(test_generators))

    results: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # A. Linear Probe Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    if cfg.eval_linear_probe:
        logger.info("\n" + "=" * 60)
        logger.info("LINEAR PROBE EVALUATION")
        logger.info("=" * 60)

        y_pred_lp, y_prob_lp = _predict_linear_probe(detector, test_feats, cfg, device)

        # Overall
        overall_lp = _evaluate_subset(y_true, y_pred_lp, y_prob_lp, "Linear Probe — Overall")

        # Per generator
        per_gen_lp: Dict[str, Dict[str, Any]] = {}
        for gen in unique_generators:
            mask = np.array([g == gen for g in test_generators])
            if mask.sum() == 0:
                continue
            gen_metrics = _evaluate_subset(
                y_true[mask], y_pred_lp[mask],
                y_prob_lp[mask] if y_prob_lp is not None else None,
                f"Linear Probe — {gen}",
            )
            per_gen_lp[gen] = gen_metrics

        results["linear_probe"] = {
            "overall": overall_lp,
            "per_generator": per_gen_lp,
        }

        # ── Save plots ───────────────────────────────────────────────────────
        _save_eval_plots(
            y_true=y_true,
            y_pred=y_pred_lp,
            y_prob=y_prob_lp,
            results_dir=results_dir,
            prefix="linear_probe",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # B. Nearest Neighbour Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    if cfg.eval_nearest_neighbor:
        logger.info("\n" + "=" * 60)
        logger.info("NEAREST NEIGHBOUR EVALUATION")
        logger.info("=" * 60)

        # Fit NN classifier on training features
        train_cache_tag = cfg.model_name.replace("/", "_").replace("-", "_")
        train_cache_path = str(
            Path(cfg.feature_cache_dir) / f"train_{train_cache_tag}.pt"
        )

        if Path(train_cache_path).exists():
            train_cache = load_cached_features(train_cache_path)
        else:
            logger.info("Train feature cache not found; extracting from %s …", cfg.train_dir)
            train_cache = prepare_feature_cache(detector, cfg, device,
                                                force_recompute=force_recompute_features)["train"]

        nn_clf = NearestNeighborClassifier()
        nn_clf.fit(train_cache["features"], train_cache["labels"])
        logger.info("NN classifier fitted on %d training embeddings.", len(train_cache["labels"]))

        # Predict on test set in batches
        batch_size = 256
        all_nn_preds: List[np.ndarray] = []
        test_feats_norm = torch.nn.functional.normalize(test_feats, p=2, dim=-1)
        for start in range(0, len(test_feats_norm), batch_size):
            batch = test_feats_norm[start : start + batch_size]
            preds_batch = nn_clf.predict(batch).numpy()
            all_nn_preds.append(preds_batch)
        y_pred_nn = np.concatenate(all_nn_preds)

        # Overall
        overall_nn = _evaluate_subset(y_true, y_pred_nn, None, "Nearest Neighbor — Overall")

        # Per generator
        per_gen_nn: Dict[str, Dict[str, Any]] = {}
        for gen in unique_generators:
            mask = np.array([g == gen for g in test_generators])
            if mask.sum() == 0:
                continue
            gen_metrics = _evaluate_subset(
                y_true[mask], y_pred_nn[mask], None, f"Nearest Neighbor — {gen}"
            )
            per_gen_nn[gen] = gen_metrics

        results["nearest_neighbor"] = {
            "overall": overall_nn,
            "per_generator": per_gen_nn,
        }

        # NN confusion matrix plot (no probabilities for ROC)
        _save_eval_plots(
            y_true=y_true,
            y_pred=y_pred_nn,
            y_prob=None,
            results_dir=results_dir,
            prefix="nearest_neighbor",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Save JSON report
    # ─────────────────────────────────────────────────────────────────────────
    report_path = results_dir / "evaluation_report.json"
    _save_json_report(results, report_path)
    logger.info("Full evaluation report saved to %s", report_path)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_eval_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    results_dir: Path,
    prefix: str,
) -> None:
    """Saves confusion matrix and (optionally) ROC curve plots."""
    from src.utils.metrics import compute_evaluation_metrics

    metrics = compute_evaluation_metrics(y_true, y_pred, y_prob)
    cm_data = metrics["confusion_matrix"]["matrix"]

    cm_path = str(results_dir / f"{prefix}_confusion_matrix.png")
    plot_confusion_matrix(
        cm=cm_data,
        save_path=cm_path,
        title=f"Confusion Matrix — {prefix.replace('_', ' ').title()}",
    )

    if y_prob is not None and len(np.unique(y_true)) > 1:
        roc_path = str(results_dir / f"{prefix}_roc_curve.png")
        plot_roc_curve(y_true=y_true, y_prob=y_prob, save_path=roc_path)


def _save_json_report(results: Dict[str, Any], path: Path) -> None:
    """Serialises the results dict to a pretty-printed JSON file.

    ``numpy`` scalars and confusion matrix arrays are converted to Python
    native types for JSON compatibility.
    """

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)
