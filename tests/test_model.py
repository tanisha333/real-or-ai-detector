"""Unit tests for Universal Fake Detector architecture and feature extraction."""

import os
import sys
import tempfile
import torch
import torch.nn.functional as F

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.detector import UniversalFakeDetector, NearestNeighborClassifier
from src.models.feature_extractor import extract_and_cache_features, load_cached_features


def test_frozen_backbone_and_trainable_probe():
    """Verify that CLIP ViT parameters are completely frozen while linear probe is trainable."""
    print("Testing frozen backbone constraints...")
    detector = UniversalFakeDetector(pretrained=True)

    backbone_frozen = all(not p.requires_grad for p in detector.backbone.parameters())
    assert backbone_frozen, "CRITICAL ERROR: Backbone parameters must be FROZEN (requires_grad=False) per CVPR 2023 paper!"

    probe_trainable = all(p.requires_grad for p in detector.linear_probe.parameters())
    assert probe_trainable, "Linear probe parameters must be trainable."

    # Total trainable parameters should be exactly feature_dim * 1 + 1 (512 * 1 + 1 = 513)
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    assert trainable_params == detector.feature_dim + 1, (
        f"Expected {detector.feature_dim + 1} trainable params, got {trainable_params}"
    )
    print(f"Verified: Backbone is 100% frozen. Only {trainable_params} probe parameters are trainable.")


def test_model_forward_and_shapes():
    """Verify input/output shapes and value bounds."""
    print("Testing forward pass and output shapes...")
    detector = UniversalFakeDetector(pretrained=True)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # 1. Feature extraction
    features = detector.extract_features(dummy_input, normalize=True)
    assert features.shape == (batch_size, detector.feature_dim), (
        f"Expected shape ({batch_size}, {detector.feature_dim}), got {features.shape}"
    )
    # Check L2 normalization: norms should be approximately 1.0
    norms = torch.norm(features, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Features must be L2 normalized."

    # 2. Forward pass (logits)
    logits = detector(dummy_input)
    assert logits.shape == (batch_size, 1), f"Expected logits shape ({batch_size}, 1), got {logits.shape}"

    # 3. Probability prediction
    probs = detector.predict_proba(dummy_input)
    assert probs.shape == (batch_size,), f"Expected probs shape ({batch_size},), got {probs.shape}"
    assert torch.all((probs >= 0.0) & (probs <= 1.0)), "Predicted probabilities must be within [0.0, 1.0]"
    print("Forward pass and shape checks passed.")


def test_probe_save_and_load():
    """Verify saving and loading of the lightweight linear probe weights."""
    print("Testing probe serialization...")
    detector = UniversalFakeDetector(pretrained=True)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Mutate probe weight slightly to ensure we test true state transfer
        with torch.no_grad():
            detector.linear_probe.weight.fill_(0.42)
            detector.linear_probe.bias.fill_(-0.25)
        
        detector.save_probe(tmp_path)
        assert os.path.getsize(tmp_path) < 50_000, "Probe file should be tiny (< 50 KB)."

        # Create fresh detector and load probe
        new_detector = UniversalFakeDetector(pretrained=True)
        new_detector.load_probe(tmp_path)

        assert torch.allclose(new_detector.linear_probe.weight, detector.linear_probe.weight)
        assert torch.allclose(new_detector.linear_probe.bias, detector.linear_probe.bias)
        print("Probe save and load passed successfully.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_nearest_neighbor_classifier():
    """Verify non-parametric nearest-neighbor cosine classification."""
    print("Testing Nearest Neighbor classifier logic...")
    nn_clf = NearestNeighborClassifier()

    # Synthetic cluster centers: Real centered near [1, 0, 0...], Fake centered near [0, 1, 0...]
    torch.manual_seed(42)
    d = 512
    real_feats = torch.randn(20, d)
    real_feats[:, 0] += 50.0  # Dominant in dimension 0
    real_labels = torch.zeros(20)

    fake_feats = torch.randn(20, d)
    fake_feats[:, 1] += 50.0  # Dominant in dimension 1
    fake_labels = torch.ones(20)

    train_feats = torch.cat([real_feats, fake_feats], dim=0)
    train_labels = torch.cat([real_labels, fake_labels], dim=0)

    nn_clf.fit(train_feats, train_labels)

    # Test query near real
    query_real = torch.randn(4, d)
    query_real[:, 0] += 50.0
    preds_real = nn_clf.predict(query_real)
    assert (preds_real == 0).all(), "Queries near real bank must be classified as 0 (Real)"

    # Test query near fake
    query_fake = torch.randn(4, d)
    query_fake[:, 1] += 50.0
    preds_fake = nn_clf.predict(query_fake)
    assert (preds_fake == 1).all(), "Queries near fake bank must be classified as 1 (Fake)"
    print("Nearest neighbor classification logic passed.")


if __name__ == "__main__":
    test_frozen_backbone_and_trainable_probe()
    test_model_forward_and_shapes()
    test_probe_save_and_load()
    test_nearest_neighbor_classifier()
    print("\nAll Stage 2 model architecture unit tests PASSED successfully!")
