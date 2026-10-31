"""Training engine for the Universal Fake Image Detector linear probe.

Implements Stage 3 of the pipeline:

  1. (Optional) Pre-extract frozen CLIP features from train / val splits and
     cache them to disk so the probe can be trained without re-running the
     expensive ViT forward pass every epoch.

  2. Train the lightweight linear probe (psi) with BCE loss, an AdamW / SGD
     optimiser, optional cosine / step LR scheduling, and early stopping.

  3. Track train / val metrics every epoch and checkpoint the best probe.

Usage
-----
This module is intended to be called from ``scripts/train.py`` but all
public functions can be used directly in a notebook or imported by tests.
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from src.models.detector import UniversalFakeDetector
from src.models.feature_extractor import extract_and_cache_features, load_cached_features
from src.data.dataset import BinaryImageDataset, create_dataloader
from src.training.config import TrainingConfig
from src.utils.metrics import compute_evaluation_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Sets random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_optimizer(probe_params, cfg: TrainingConfig):
    """Instantiates the configured optimiser over the linear probe parameters only."""
    if cfg.optimizer.lower() == "adamw":
        return AdamW(probe_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == "sgd":
        return SGD(
            probe_params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Choose 'adamw' or 'sgd'.")


def _build_scheduler(optimizer, cfg: TrainingConfig, steps_per_epoch: int):
    """Builds a warm-up + main LR scheduler.

    Returns a tuple (warmup_scheduler | None, main_scheduler | None).
    """
    warmup_sched = None
    main_sched = None

    if cfg.warmup_epochs > 0:
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_sched = LambdaLR(optimizer, lr_lambda=lr_lambda)

    remaining_epochs = max(1, cfg.num_epochs - cfg.warmup_epochs)
    if cfg.scheduler.lower() == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
    elif cfg.scheduler.lower() == "step":
        main_sched = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma)
    # 'none' → main_sched stays None

    return warmup_sched, main_sched


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_feature_cache(
    detector: UniversalFakeDetector,
    cfg: TrainingConfig,
    device: torch.device,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Ensures pre-extracted CLIP features exist on disk for train and val splits.

    If the .pt cache files already exist they are loaded; otherwise features
    are extracted from the raw images and saved.

    Args:
        detector:        Initialized UniversalFakeDetector (backbone frozen).
        cfg:             TrainingConfig instance with all path / batch settings.
        device:          Torch device for the extraction forward pass.
        force_recompute: Re-extract even if cache files already exist.

    Returns:
        Dict with keys ``'train'`` and ``'val'``, each mapping to a feature
        cache dictionary (see ``extract_and_cache_features``).
    """
    cache_dir = Path(cfg.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Derive a short name from the model string for cache file naming
    model_tag = cfg.model_name.replace("/", "_").replace("-", "_")

    splits = {
        "train": cfg.train_dir,
        "val": cfg.val_dir,
    }
    caches: Dict[str, Dict[str, Any]] = {}

    for split, data_dir in splits.items():
        cache_path = str(cache_dir / f"{split}_{model_tag}.pt")

        if not force_recompute and os.path.exists(cache_path):
            logger.info("Loading cached %s features from %s", split, cache_path)
            caches[split] = load_cached_features(cache_path)
        else:
            logger.info("Extracting %s features from %s …", split, data_dir)
            dataset = BinaryImageDataset(root_dir=data_dir)
            if len(dataset) == 0:
                raise RuntimeError(
                    f"No images found in '{data_dir}'. "
                    "Ensure the directory has 'real/' and 'fake/' sub-folders."
                )
            counts = dataset.get_class_counts()
            logger.info("  %s split: %d real, %d fake, %d total", split,
                        counts["real"], counts["fake"], counts["total"])

            dl = create_dataloader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
            caches[split] = extract_and_cache_features(detector, dl, cache_path, device)

    return caches


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch routines
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    detector: UniversalFakeDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: TrainingConfig,
    warmup_sched=None,
    global_step: int = 0,
    train: bool = True,
) -> Tuple[float, float, float, int]:
    """Runs one epoch over a feature DataLoader.

    Args:
        detector:    The detector (probe is in train or eval mode as appropriate).
        loader:      DataLoader of (features, labels) tensors.
        criterion:   Loss function (BCEWithLogitsLoss).
        optimizer:   Optimizer (None during validation).
        device:      Compute device.
        cfg:         Training config.
        warmup_sched: Warm-up LR scheduler (stepped per batch during warm-up).
        global_step: Current global training step count.
        train:       True → gradient updates; False → inference only.

    Returns:
        Tuple of (avg_loss, accuracy, avg_prob_fake, updated_global_step).
    """
    if train:
        detector.train()
        # Always keep backbone frozen even in training mode
        detector.backbone.eval()
    else:
        detector.eval()

    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []
    all_probs: list = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch_idx, (feats, labels) in enumerate(loader):
            feats = feats.to(device)
            labels = labels.to(device).unsqueeze(1)  # (B, 1) to match logit shape

            logits = detector.forward_features(feats)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step warm-up scheduler per batch
                if warmup_sched is not None:
                    warmup_sched.step()

                global_step += 1

                if global_step % cfg.log_every_n_steps == 0:
                    logger.debug(
                        "  step %d | batch_loss=%.4f | lr=%.6f",
                        global_step,
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )

            total_loss += loss.item()
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu()
            preds = (probs >= cfg.threshold).long()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.squeeze(1).long().cpu())

    all_preds_t = torch.cat(all_preds).numpy()
    all_labels_t = torch.cat(all_labels).numpy()
    all_probs_t = torch.cat(all_probs).numpy()

    avg_loss = total_loss / max(1, len(loader))
    metrics = compute_evaluation_metrics(all_labels_t, all_preds_t, all_probs_t)
    accuracy = metrics["accuracy"]
    avg_prob_fake = float(all_probs_t.mean())

    return avg_loss, accuracy, avg_prob_fake, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: TrainingConfig,
    detector: Optional[UniversalFakeDetector] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Runs the full Stage 3 linear probe training pipeline.

    Steps
    -----
    1. Set seeds and select device.
    2. Build (or accept) the detector; verify backbone is frozen.
    3. Pre-extract / load CLIP feature caches for train + val.
    4. Wrap cached tensors in TensorDatasets / DataLoaders.
    5. Configure optimiser and LR schedulers.
    6. Run training loop with early stopping and best-model checkpointing.
    7. Return a history dict with per-epoch metrics.

    Args:
        cfg:      TrainingConfig with all hyperparameters and paths.
        detector: Optional pre-built detector. Created from cfg if not provided.
        device:   Target device. Auto-detected if not provided.

    Returns:
        history: Dict with keys ``'train'``, ``'val'``, and ``'best_epoch'``,
                 each containing lists of per-epoch metrics.
    """
    set_seed(cfg.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ── 1. Build model ───────────────────────────────────────────────────────
    if detector is None:
        logger.info("Initializing UniversalFakeDetector (%s) …", cfg.model_name)
        detector = UniversalFakeDetector(
            model_name=cfg.model_name,
            dropout=cfg.dropout,
        )

    # Safety check: backbone must be frozen
    if any(p.requires_grad for p in detector.backbone.parameters()):
        raise RuntimeError(
            "Backbone parameters are not frozen! "
            "All CLIP ViT weights must have requires_grad=False."
        )

    detector.to(device)
    trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    logger.info("Trainable probe parameters: %d", trainable)

    # ── 2. Prepare feature caches ────────────────────────────────────────────
    if cfg.use_feature_cache:
        logger.info("Preparing CLIP feature caches …")
        caches = prepare_feature_cache(detector, cfg, device)
        train_feats = caches["train"]["features"]
        train_labels = caches["train"]["labels"]
        val_feats = caches["val"]["features"]
        val_labels = caches["val"]["labels"]
    else:
        # Extract features on-the-fly during training (slower but avoids disk I/O)
        logger.info("Extracting features on-the-fly (use_feature_cache=False) …")
        train_ds = BinaryImageDataset(root_dir=cfg.train_dir)
        val_ds = BinaryImageDataset(root_dir=cfg.val_dir)
        train_dl_raw = create_dataloader(train_ds, batch_size=cfg.batch_size,
                                         shuffle=False, num_workers=cfg.num_workers)
        val_dl_raw = create_dataloader(val_ds, batch_size=cfg.batch_size,
                                       shuffle=False, num_workers=cfg.num_workers)

        def _collect(dl):
            feats_list, labels_list = [], []
            with torch.no_grad():
                for imgs, lbls, _ in dl:
                    feats_list.append(detector.extract_features(imgs.to(device)).cpu())
                    labels_list.append(lbls)
            return torch.cat(feats_list), torch.cat(labels_list)

        train_feats, train_labels = _collect(train_dl_raw)
        val_feats, val_labels = _collect(val_dl_raw)

    logger.info(
        "Feature shapes — Train: %s | Val: %s",
        tuple(train_feats.shape), tuple(val_feats.shape),
    )

    # ── 3. Build TensorDataLoaders ───────────────────────────────────────────
    train_tensor_ds = TensorDataset(train_feats, train_labels)
    val_tensor_ds = TensorDataset(val_feats, val_labels)

    train_loader = DataLoader(
        train_tensor_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_tensor_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── 4. Loss / Optimiser / Schedulers ─────────────────────────────────────
    # Compute positive class weight to handle class imbalance
    n_real = int((train_labels == 0).sum().item())
    n_fake = int((train_labels == 1).sum().item())
    if n_fake > 0 and n_real > 0:
        pos_weight = torch.tensor([n_real / n_fake], device=device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = _build_optimizer(detector.linear_probe.parameters(), cfg)

    steps_per_epoch = len(train_loader)
    warmup_sched, main_sched = _build_scheduler(optimizer, cfg, steps_per_epoch)

    # ── 5. Training loop ─────────────────────────────────────────────────────
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = str(checkpoint_dir / "best_probe.pt")

    history: Dict[str, Any] = {
        "train": {"loss": [], "accuracy": [], "lr": []},
        "val": {"loss": [], "accuracy": []},
        "best_epoch": 0,
        "best_val_acc": 0.0,
    }

    patience_counter = 0
    global_step = 0
    warmup_total_steps = cfg.warmup_epochs * steps_per_epoch

    logger.info("=" * 60)
    logger.info("Starting training for %d epochs", cfg.num_epochs)
    logger.info("  Train size: %d | Val size: %d", len(train_tensor_ds), len(val_tensor_ds))
    logger.info("  Optimizer : %s | LR: %.1e | Scheduler: %s",
                cfg.optimizer, cfg.learning_rate, cfg.scheduler)
    logger.info("  Class balance — real: %d, fake: %d", n_real, n_fake)
    logger.info("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        # ── train ────────────────────────────────────────────────────────────
        train_loss, train_acc, _, global_step = _run_epoch(
            detector, train_loader, criterion, optimizer, device, cfg,
            warmup_sched=(warmup_sched if global_step < warmup_total_steps else None),
            global_step=global_step,
            train=True,
        )

        # ── val ──────────────────────────────────────────────────────────────
        val_loss, val_acc, _, _ = _run_epoch(
            detector, val_loader, criterion, None, device, cfg,
            global_step=global_step, train=False,
        )

        # ── main LR scheduler step (per epoch, after warm-up) ───────────────
        current_lr = optimizer.param_groups[0]["lr"]
        if main_sched is not None and epoch > cfg.warmup_epochs:
            main_sched.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f  train_acc=%.2f%%  "
            "val_loss=%.4f  val_acc=%.2f%%  lr=%.2e  (%.1fs)",
            epoch, cfg.num_epochs,
            train_loss, train_acc * 100,
            val_loss, val_acc * 100,
            current_lr, elapsed,
        )

        # Record history
        history["train"]["loss"].append(train_loss)
        history["train"]["accuracy"].append(train_acc)
        history["train"]["lr"].append(current_lr)
        history["val"]["loss"].append(val_loss)
        history["val"]["accuracy"].append(val_acc)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            patience_counter = 0
            detector.save_probe(best_checkpoint_path)
            logger.info("  [best] New best val_acc=%.2f%% — probe saved to %s",
                        val_acc * 100, best_checkpoint_path)
        else:
            patience_counter += 1
            if not cfg.save_best_only:
                ckpt = str(checkpoint_dir / f"probe_epoch_{epoch:03d}.pt")
                detector.save_probe(ckpt)

        # ── Early stopping ───────────────────────────────────────────────────
        if cfg.early_stopping_patience > 0 and patience_counter >= cfg.early_stopping_patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                cfg.early_stopping_patience,
            )
            break

    logger.info("=" * 60)
    logger.info(
        "Training complete. Best val_acc=%.2f%% at epoch %d.",
        history["best_val_acc"] * 100, history["best_epoch"],
    )
    logger.info("Best probe saved at: %s", best_checkpoint_path)

    # Reload the best probe weights into the detector before returning
    if os.path.exists(best_checkpoint_path):
        detector.load_probe(best_checkpoint_path, map_location=str(device))

    return history
