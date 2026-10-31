"""Training configuration dataclass for the Universal Fake Detector.

All hyperparameters for Stage 3 training and evaluation are centralised here
so that train.py and evaluate.py share a single source of truth.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Hyperparameters and path settings for training the linear probe.

    Naming follows the conventions of Ojha et al. (CVPR 2023) wherever
    possible so comparisons with the paper are straightforward.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "openai/clip-vit-large-patch14"
    """CLIP variant to use as the frozen backbone.
    The paper's best results use ViT-L/14; ViT-B/32 trains faster for quick runs."""

    dropout: float = 0.0
    """Dropout rate before the linear probe (0.0 per paper, configurable for ablations)."""

    # ── Data paths ───────────────────────────────────────────────────────────
    train_dir: str = "data/raw/train"
    """Root directory with 'real/' and 'fake/' sub-folders for training."""

    val_dir: str = "data/raw/val"
    """Root directory with 'real/' and 'fake/' sub-folders for validation."""

    test_dir: str = "data/raw/test"
    """Root directory with 'real/' and per-generator sub-folders for test evaluation."""

    feature_cache_dir: str = "data/features"
    """Directory where pre-extracted CLIP feature tensors (.pt files) are stored."""

    checkpoint_dir: str = "checkpoints"
    """Directory where trained probe checkpoints are saved."""

    results_dir: str = "results"
    """Directory where evaluation reports and plots are saved."""

    # ── DataLoader ───────────────────────────────────────────────────────────
    batch_size: int = 64
    """Batch size for feature extraction DataLoaders."""

    num_workers: int = 0
    """DataLoader worker count. 0 = main process only (safe on Windows)."""

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer: str = "adamw"
    """Optimiser type: 'adamw' (default) or 'sgd'."""

    learning_rate: float = 1e-3
    """Initial learning rate. AdamW default used in linear probe literature."""

    weight_decay: float = 1e-4
    """L2 regularisation coefficient."""

    momentum: float = 0.9
    """Momentum for SGD (ignored when optimizer='adamw')."""

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler: str = "cosine"
    """LR scheduler: 'cosine', 'step', or 'none'."""

    warmup_epochs: int = 1
    """Number of linear warm-up epochs before the main schedule."""

    step_size: int = 10
    """Step size (epochs) for StepLR (ignored unless scheduler='step')."""

    step_gamma: float = 0.5
    """Multiplicative LR decay for StepLR (ignored unless scheduler='step')."""

    # ── Training loop ────────────────────────────────────────────────────────
    num_epochs: int = 20
    """Maximum number of training epochs."""

    early_stopping_patience: int = 5
    """Stop training if val accuracy does not improve for this many epochs (-1 = disabled)."""

    use_feature_cache: bool = True
    """If True, pre-extract and cache all CLIP features before the training loop
    (much faster when iterating on probe hyperparameters)."""

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_linear_probe: bool = True
    """Run final evaluation with the trained linear probe."""

    eval_nearest_neighbor: bool = True
    """Run final evaluation with the non-parametric NN classifier."""

    threshold: float = 0.5
    """Decision threshold for converting sigmoid probabilities to binary labels."""

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every_n_steps: int = 10
    """Print training metrics every N gradient steps."""

    save_best_only: bool = True
    """Save only the best checkpoint (by val accuracy) rather than every epoch."""
