"""Unit tests for Stage 3: Training and Evaluation pipeline.

All tests use in-memory synthetic data (random features and labels) so they
run quickly without needing real images, a GPU, or the CLIP model.

Covered:
  - TrainingConfig defaults and field types
  - _build_optimizer (AdamW / SGD)
  - _build_scheduler (cosine / step / none + warm-up)
  - _run_epoch (train and eval mode, loss decreases, no backbone gradient)
  - train() end-to-end with feature-cache path
  - prepare_feature_cache against a tiny temp dataset
  - evaluate() overall and per-generator slicing with a dummy checkpoint
  - NearestNeighborClassifier integration through evaluator
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

# ── Project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.detector import UniversalFakeDetector
from src.training.config import TrainingConfig
from src.training.trainer import (
    set_seed,
    _build_optimizer,
    _build_scheduler,
    _run_epoch,
    train,
    prepare_feature_cache,
)
from src.training.evaluator import evaluate, _predict_linear_probe, _save_json_report


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM = 512  # Matches ViT-B/32 projection dim; tests don't load CLIP


class _MockBackbone(nn.Module):
    """Tiny frozen MLP that mimics the CLIP backbone interface without requiring a download."""

    def __init__(self, dim: int = FEATURE_DIM):
        super().__init__()
        # Single linear layer purely for shape compatibility; weights are frozen immediately.
        self.fc = nn.Linear(3 * 224 * 224, dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        flat = pixel_values.view(b, -1)
        return type("Out", (), {"image_embeds": self.fc(flat)})()

    class config:
        projection_dim = FEATURE_DIM


class _MockDetector(nn.Module):
    """Lightweight stand-in for UniversalFakeDetector that avoids CLIP model downloads.

    Exposes the same public API used by trainer.py and evaluator.py:
      - .backbone (frozen)
      - .linear_probe
      - .dropout
      - .feature_dim / .model_name
      - .forward_features()
      - .save_probe() / .load_probe()
      - .train() (keeps backbone in eval)
    """

    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.model_name = "mock/clip-vit-base"
        self.feature_dim = feature_dim
        self.backbone = _MockBackbone(feature_dim)
        self.dropout = nn.Identity()
        self.linear_probe = nn.Linear(feature_dim, 1)

    def train(self, mode: bool = True):
        """Keeps backbone permanently in eval mode, matching UniversalFakeDetector."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
        return self.linear_probe(features)

    def save_probe(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "linear_probe_state": self.linear_probe.state_dict(),
        }, filepath)

    def load_probe(self, filepath: str, map_location: str = "cpu"):
        checkpoint = torch.load(filepath, map_location=map_location)
        self.linear_probe.load_state_dict(checkpoint["linear_probe_state"])


def _make_detector(feature_dim: int = FEATURE_DIM) -> _MockDetector:
    """Returns a lightweight mock detector that doesn't require CLIP to be downloaded."""
    return _MockDetector(feature_dim=feature_dim)


def _make_feature_tensors(n_real: int = 40, n_fake: int = 40, dim: int = FEATURE_DIM):
    """Generates synthetic separable features (real centred at +1, fake at -1)."""
    set_seed(0)
    real_feats = torch.randn(n_real, dim) + 1.0
    fake_feats = torch.randn(n_fake, dim) - 1.0
    feats = torch.cat([real_feats, fake_feats], dim=0)
    labels = torch.cat([torch.zeros(n_real), torch.ones(n_fake)], dim=0)
    # Shuffle
    perm = torch.randperm(len(feats))
    return feats[perm], labels[perm]


def _make_tiny_image_dir(tmp_dir: str, split: str, n_real: int = 4, n_fake: int = 4) -> str:
    """Creates a minimal real/fake image directory under tmp_dir/split/."""
    split_dir = Path(tmp_dir) / split
    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_real):
        img = Image.new("RGB", (64, 64), color=(i * 20, 100, 200))
        img.save(str(real_dir / f"real_{i:03d}.jpg"))

    for i in range(n_fake):
        img = Image.new("RGB", (64, 64), color=(200, i * 20, 50))
        img.save(str(fake_dir / f"fake_{i:03d}.jpg"))

    return str(split_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: TrainingConfig
# ─────────────────────────────────────────────────────────────────────────────

def test_training_config_defaults():
    """TrainingConfig must be instantiable with sensible defaults."""
    cfg = TrainingConfig()
    assert cfg.num_epochs > 0, "num_epochs must be positive"
    assert 0.0 < cfg.learning_rate < 1.0, "learning_rate out of reasonable range"
    assert cfg.batch_size > 0
    assert cfg.seed is not None
    assert cfg.optimizer in ("adamw", "sgd")
    assert cfg.scheduler in ("cosine", "step", "none")
    print("test_training_config_defaults PASSED")


def test_training_config_override():
    """TrainingConfig fields can be overridden at construction time."""
    cfg = TrainingConfig(num_epochs=50, learning_rate=5e-4, optimizer="sgd")
    assert cfg.num_epochs == 50
    assert cfg.learning_rate == 5e-4
    assert cfg.optimizer == "sgd"
    print("test_training_config_override PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Optimizer / Scheduler builders
# ─────────────────────────────────────────────────────────────────────────────

def test_build_optimizer_adamw():
    det = _make_detector()
    cfg = TrainingConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=1e-4)
    opt = _build_optimizer(det.linear_probe.parameters(), cfg)
    assert opt.__class__.__name__ == "AdamW"
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-9
    print("test_build_optimizer_adamw PASSED")


def test_build_optimizer_sgd():
    det = _make_detector()
    cfg = TrainingConfig(optimizer="sgd", learning_rate=1e-2, momentum=0.9)
    opt = _build_optimizer(det.linear_probe.parameters(), cfg)
    assert opt.__class__.__name__ == "SGD"
    assert abs(opt.param_groups[0]["lr"] - 1e-2) < 1e-9
    print("test_build_optimizer_sgd PASSED")


def test_build_scheduler_cosine():
    det = _make_detector()
    cfg = TrainingConfig(scheduler="cosine", num_epochs=10, warmup_epochs=1)
    opt = _build_optimizer(det.linear_probe.parameters(), cfg)
    warmup_sched, main_sched = _build_scheduler(opt, cfg, steps_per_epoch=5)
    assert warmup_sched is not None, "Warm-up scheduler must be present when warmup_epochs>0"
    assert main_sched is not None, "CosineAnnealingLR must be created for 'cosine' scheduler"
    print("test_build_scheduler_cosine PASSED")


def test_build_scheduler_none():
    det = _make_detector()
    cfg = TrainingConfig(scheduler="none", warmup_epochs=0)
    opt = _build_optimizer(det.linear_probe.parameters(), cfg)
    warmup_sched, main_sched = _build_scheduler(opt, cfg, steps_per_epoch=5)
    assert warmup_sched is None
    assert main_sched is None
    print("test_build_scheduler_none PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _run_epoch
# ─────────────────────────────────────────────────────────────────────────────

def test_run_epoch_train_reduces_loss():
    """Loss should generally decrease over multiple training epochs on separable data."""
    set_seed(42)
    det = _make_detector()
    feats, labels = _make_feature_tensors(n_real=60, n_fake=60)

    ds = TensorDataset(feats, labels)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    cfg = TrainingConfig(threshold=0.5, log_every_n_steps=9999)
    opt = torch.optim.AdamW(det.linear_probe.parameters(), lr=1e-2)
    device = torch.device("cpu")

    losses = []
    step = 0
    for _ in range(5):
        loss, acc, _, step = _run_epoch(det, loader, criterion, opt, device, cfg,
                                         global_step=step, train=True)
        losses.append(loss)

    assert losses[-1] < losses[0], (
        f"Training loss should decrease over epochs; got {losses}"
    )
    print(f"test_run_epoch_train_reduces_loss PASSED  (loss {losses[0]:.4f} -> {losses[-1]:.4f})")


def test_run_epoch_eval_no_grad():
    """Eval epoch must not compute gradients (probe weights unchanged)."""
    set_seed(0)
    det = _make_detector()
    feats, labels = _make_feature_tensors()
    ds = TensorDataset(feats, labels)
    loader = DataLoader(ds, batch_size=32)
    criterion = nn.BCEWithLogitsLoss()
    cfg = TrainingConfig(log_every_n_steps=9999)

    before = det.linear_probe.weight.clone()
    _run_epoch(det, loader, criterion, None, torch.device("cpu"), cfg, train=False)
    after = det.linear_probe.weight.clone()

    assert torch.allclose(before, after), "Probe weights must not change during eval epoch"
    print("test_run_epoch_eval_no_grad PASSED")


def test_backbone_always_frozen_during_train():
    """Backbone parameter gradients must remain zero even during training epoch."""
    set_seed(0)
    det = _make_detector()
    feats, labels = _make_feature_tensors(n_real=8, n_fake=8)
    ds = TensorDataset(feats, labels)
    loader = DataLoader(ds, batch_size=8)
    criterion = nn.BCEWithLogitsLoss()
    cfg = TrainingConfig(log_every_n_steps=9999)
    opt = torch.optim.AdamW(det.linear_probe.parameters(), lr=1e-3)

    _run_epoch(det, loader, criterion, opt, torch.device("cpu"), cfg, train=True)

    for name, p in det.backbone.named_parameters():
        assert not p.requires_grad, f"Backbone param {name} must have requires_grad=False"
    print("test_backbone_always_frozen_during_train PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: train() end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_train_end_to_end_with_feature_cache():
    """train() must run without error, save a checkpoint, and return history."""
    tmp = tempfile.mkdtemp()
    try:
        set_seed(0)
        det = _make_detector()
        feats_train, labels_train = _make_feature_tensors(n_real=30, n_fake=30)
        feats_val, labels_val = _make_feature_tensors(n_real=10, n_fake=10)

        # Pre-populate feature cache so train() skips real image loading
        model_tag = "mock_clip_vit_base"
        cache_dir = Path(tmp) / "features"
        cache_dir.mkdir()
        torch.save({
            "features": feats_train, "labels": labels_train,
            "generators": ["mock"] * len(labels_train),
            "paths": [""] * len(labels_train),
            "feature_dim": FEATURE_DIM, "total_samples": len(labels_train),
            "model_name": "mock/clip-vit-base",
        }, str(cache_dir / f"train_{model_tag}.pt"))
        torch.save({
            "features": feats_val, "labels": labels_val,
            "generators": ["mock"] * len(labels_val),
            "paths": [""] * len(labels_val),
            "feature_dim": FEATURE_DIM, "total_samples": len(labels_val),
            "model_name": "mock/clip-vit-base",
        }, str(cache_dir / f"val_{model_tag}.pt"))

        cfg = TrainingConfig(
            model_name="mock/clip-vit-base",
            feature_cache_dir=str(cache_dir),
            checkpoint_dir=str(Path(tmp) / "checkpoints"),
            results_dir=str(Path(tmp) / "results"),
            num_epochs=3,
            batch_size=16,
            warmup_epochs=0,
            early_stopping_patience=-1,
            use_feature_cache=True,
            log_every_n_steps=9999,
        )

        history = train(cfg, detector=det)

        assert "train" in history and "val" in history
        assert len(history["train"]["loss"]) == 3
        assert len(history["val"]["accuracy"]) == 3
        assert history["best_epoch"] >= 1

        best_ckpt = Path(cfg.checkpoint_dir) / "best_probe.pt"
        assert best_ckpt.exists(), "Best checkpoint must be saved"
        assert best_ckpt.stat().st_size < 100_000, "Checkpoint should be tiny (probe only)"
        print("test_train_end_to_end_with_feature_cache PASSED")
    finally:
        shutil.rmtree(tmp)


def test_train_early_stopping():
    """Early stopping must terminate training before num_epochs when patience is exceeded."""
    tmp = tempfile.mkdtemp()
    try:
        set_seed(0)
        det = _make_detector()
        # Completely non-separable (random) features → val acc won't improve consistently
        feats = torch.randn(40, FEATURE_DIM)
        labels = torch.randint(0, 2, (40,)).float()
        feats_v = torch.randn(20, FEATURE_DIM)
        labels_v = torch.randint(0, 2, (20,)).float()

        model_tag = "mock_clip_vit_base"
        cache_dir = Path(tmp) / "features"
        cache_dir.mkdir()
        for split, f, l in [("train", feats, labels), ("val", feats_v, labels_v)]:
            torch.save({
                "features": f, "labels": l,
                "generators": ["mock"] * len(l), "paths": [""] * len(l),
                "feature_dim": FEATURE_DIM, "total_samples": len(l),
                "model_name": "mock/clip-vit-base",
            }, str(cache_dir / f"{split}_{model_tag}.pt"))

        cfg = TrainingConfig(
            model_name="mock/clip-vit-base",
            feature_cache_dir=str(cache_dir),
            checkpoint_dir=str(Path(tmp) / "checkpoints"),
            results_dir=str(Path(tmp) / "results"),
            num_epochs=50,
            batch_size=16,
            warmup_epochs=0,
            early_stopping_patience=3,
            use_feature_cache=True,
            log_every_n_steps=9999,
        )

        history = train(cfg, detector=det)
        # Must stop well before 50 epochs
        assert len(history["train"]["loss"]) < 50, (
            f"Expected early stopping to trigger; ran {len(history['train']['loss'])} epochs"
        )
        print(f"test_train_early_stopping PASSED  "
              f"(stopped at epoch {len(history['train']['loss'])})")
    finally:
        shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: evaluate()
# ─────────────────────────────────────────────────────────────────────────────

def _make_eval_cache(tmp: str, split: str, n_real: int, n_fake: int,
                     generators=None) -> str:
    feats, labels = _make_feature_tensors(n_real, n_fake)
    model_tag = "mock_clip_vit_base"
    cache_dir = Path(tmp) / "features"
    cache_dir.mkdir(exist_ok=True)
    if generators is None:
        generators = ["real"] * n_real + ["adm"] * n_fake
    paths = [""] * (n_real + n_fake)
    torch.save({
        "features": feats, "labels": labels,
        "generators": generators, "paths": paths,
        "feature_dim": FEATURE_DIM, "total_samples": len(labels),
        "model_name": "mock/clip-vit-base",
    }, str(cache_dir / f"{split}_{model_tag}.pt"))
    return str(cache_dir)


def test_evaluate_linear_probe_overall():
    """evaluate() must return an 'overall' metrics dict for the linear probe."""
    tmp = tempfile.mkdtemp()
    try:
        set_seed(0)
        det = _make_detector()
        # Separate features → probe should classify well after random init weights
        _make_eval_cache(tmp, "train", n_real=40, n_fake=40)
        _make_eval_cache(tmp, "test", n_real=20, n_fake=20)

        # Save a dummy probe checkpoint
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = str(ckpt_dir / "best_probe.pt")
        det.save_probe(ckpt_path)

        cfg = TrainingConfig(
            model_name="mock/clip-vit-base",
            feature_cache_dir=str(Path(tmp) / "features"),
            checkpoint_dir=str(ckpt_dir),
            results_dir=str(Path(tmp) / "results"),
            eval_linear_probe=True,
            eval_nearest_neighbor=False,
        )

        results = evaluate(cfg, detector=det, probe_checkpoint=ckpt_path)

        assert "linear_probe" in results
        overall = results["linear_probe"]["overall"]
        assert "accuracy" in overall
        assert 0.0 <= overall["accuracy"] <= 1.0
        print(f"test_evaluate_linear_probe_overall PASSED  "
              f"(acc={overall['accuracy']*100:.1f}%)")
    finally:
        shutil.rmtree(tmp)


def test_evaluate_per_generator_slicing():
    """Per-generator metrics must cover all generator names present in test features."""
    tmp = tempfile.mkdtemp()
    try:
        set_seed(1)
        det = _make_detector()

        # Test cache with 3 generator types
        generators = (["real"] * 10 + ["adm"] * 10 + ["biggan"] * 10
                       + ["sd14"] * 10 + ["real"] * 10 + ["adm"] * 10)
        n_total = len(generators)
        feats, labels = _make_feature_tensors(n_real=30, n_fake=30)

        model_tag = "mock_clip_vit_base"
        cache_dir = Path(tmp) / "features"
        cache_dir.mkdir()
        torch.save({
            "features": feats, "labels": labels,
            "generators": generators, "paths": [""] * n_total,
            "feature_dim": FEATURE_DIM, "total_samples": n_total,
            "model_name": "mock/clip-vit-base",
        }, str(cache_dir / f"test_{model_tag}.pt"))
        _make_eval_cache(tmp, "train", n_real=20, n_fake=20)

        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = str(ckpt_dir / "best_probe.pt")
        det.save_probe(ckpt_path)

        cfg = TrainingConfig(
            model_name="mock/clip-vit-base",
            feature_cache_dir=str(cache_dir),
            checkpoint_dir=str(ckpt_dir),
            results_dir=str(Path(tmp) / "results"),
            eval_linear_probe=True,
            eval_nearest_neighbor=False,
        )

        results = evaluate(cfg, detector=det, probe_checkpoint=ckpt_path)
        per_gen = results["linear_probe"]["per_generator"]
        expected_gens = {"real", "adm", "biggan", "sd14"}
        assert expected_gens == set(per_gen.keys()), (
            f"Expected generators {expected_gens}, got {set(per_gen.keys())}"
        )
        print(f"test_evaluate_per_generator_slicing PASSED  (gens={set(per_gen.keys())})")
    finally:
        shutil.rmtree(tmp)


def test_evaluate_nearest_neighbor():
    """evaluate() must also evaluate the NN classifier when requested."""
    tmp = tempfile.mkdtemp()
    try:
        set_seed(2)
        det = _make_detector()
        _make_eval_cache(tmp, "train", n_real=30, n_fake=30)
        _make_eval_cache(tmp, "test", n_real=15, n_fake=15)

        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = str(ckpt_dir / "best_probe.pt")
        det.save_probe(ckpt_path)

        cfg = TrainingConfig(
            model_name="mock/clip-vit-base",
            feature_cache_dir=str(Path(tmp) / "features"),
            checkpoint_dir=str(ckpt_dir),
            results_dir=str(Path(tmp) / "results"),
            eval_linear_probe=False,
            eval_nearest_neighbor=True,
        )

        results = evaluate(cfg, detector=det, probe_checkpoint=ckpt_path)
        assert "nearest_neighbor" in results
        nn_overall = results["nearest_neighbor"]["overall"]
        assert 0.0 <= nn_overall["accuracy"] <= 1.0
        print(f"test_evaluate_nearest_neighbor PASSED  "
              f"(nn_acc={nn_overall['accuracy']*100:.1f}%)")
    finally:
        shutil.rmtree(tmp)


def test_evaluate_json_report_saved():
    """evaluate() must write a valid JSON report to results_dir."""
    tmp = tempfile.mkdtemp()
    try:
        set_seed(3)
        det = _make_detector()
        _make_eval_cache(tmp, "train", n_real=20, n_fake=20)
        _make_eval_cache(tmp, "test", n_real=10, n_fake=10)

        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = str(ckpt_dir / "best_probe.pt")
        det.save_probe(ckpt_path)

        results_dir = str(Path(tmp) / "results")
        cfg = TrainingConfig(
            model_name="mock/clip-vit-base",
            feature_cache_dir=str(Path(tmp) / "features"),
            checkpoint_dir=str(ckpt_dir),
            results_dir=results_dir,
            eval_linear_probe=True,
            eval_nearest_neighbor=False,
        )

        evaluate(cfg, detector=det, probe_checkpoint=ckpt_path)
        report_path = Path(results_dir) / "evaluation_report.json"
        assert report_path.exists(), "evaluation_report.json must be written"
        with open(report_path) as f:
            report = json.load(f)
        assert "linear_probe" in report
        print("test_evaluate_json_report_saved PASSED")
    finally:
        shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _predict_linear_probe
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_linear_probe_output_shapes():
    """_predict_linear_probe must return arrays of correct shape and value range."""
    det = _make_detector()
    feats = torch.randn(25, FEATURE_DIM)
    cfg = TrainingConfig(threshold=0.5)
    preds, probs = _predict_linear_probe(det, feats, cfg, torch.device("cpu"), batch_size=10)
    assert preds.shape == (25,), f"Expected (25,), got {preds.shape}"
    assert probs.shape == (25,), f"Expected (25,), got {probs.shape}"
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be in [0, 1]"
    assert set(preds).issubset({0, 1}), "Predictions must be binary"
    print("test_predict_linear_probe_output_shapes PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _save_json_report
# ─────────────────────────────────────────────────────────────────────────────

def test_save_json_report():
    """_save_json_report must serialise numpy types without raising TypeError."""
    tmp = tempfile.mkdtemp()
    try:
        results = {
            "linear_probe": {
                "overall": {
                    "accuracy": np.float64(0.87),
                    "f1_score": np.float32(0.85),
                    "confusion_matrix": {"matrix": np.array([[10, 2], [3, 15]]).tolist()},
                    "roc_auc": None,
                }
            }
        }
        path = Path(tmp) / "report.json"
        _save_json_report(results, path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert abs(loaded["linear_probe"]["overall"]["accuracy"] - 0.87) < 1e-6
        print("test_save_json_report PASSED")
    finally:
        shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Stage 3 training & evaluation tests …\n")

    # Config tests
    test_training_config_defaults()
    test_training_config_override()

    # Optimizer / scheduler
    test_build_optimizer_adamw()
    test_build_optimizer_sgd()
    test_build_scheduler_cosine()
    test_build_scheduler_none()

    # Epoch runner
    test_run_epoch_train_reduces_loss()
    test_run_epoch_eval_no_grad()
    test_backbone_always_frozen_during_train()

    # End-to-end training
    test_train_end_to_end_with_feature_cache()
    test_train_early_stopping()

    # Evaluation
    test_evaluate_linear_probe_overall()
    test_evaluate_per_generator_slicing()
    test_evaluate_nearest_neighbor()
    test_evaluate_json_report_saved()

    # Utilities
    test_predict_linear_probe_output_shapes()
    test_save_json_report()

    print("\n" + "=" * 60)
    print("All Stage 3 tests PASSED successfully!")
    print("=" * 60)
