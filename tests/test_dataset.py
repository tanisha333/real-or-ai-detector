"""Unit tests for dataset loading, transforms, and Fourier spectrum processing."""

import os
import sys
import shutil
import tempfile

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image
from src.data.preprocessing import get_clip_transform, compute_fourier_spectrum
from src.data.dataset import BinaryImageDataset


def test_clip_transform():
    """Verify that CLIP transform outputs expected tensor shape and normalization."""
    # Create arbitrary non-square image
    test_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))
    transform = get_clip_transform(target_size=224)
    tensor = transform(test_img)

    assert isinstance(tensor, torch.Tensor), "Output should be a torch.Tensor"
    assert tensor.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32, "Expected float32 tensor"
    print("test_clip_transform passed.")


def test_fourier_spectrum():
    """Verify that Fourier spectrum calculation produces valid log-magnitude maps."""
    test_img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
    spectrum = compute_fourier_spectrum(test_img)

    assert isinstance(spectrum, np.ndarray), "Output must be a numpy ndarray"
    assert spectrum.shape == (128, 128), f"Expected shape (128, 128), got {spectrum.shape}"
    assert spectrum.dtype == np.uint8, "Expected uint8 spectrum image"
    assert spectrum.min() >= 0 and spectrum.max() <= 255, "Values must be within [0, 255]"
    print("test_fourier_spectrum passed.")


def test_binary_image_dataset():
    """Verify BinaryImageDataset directory parsing and item access."""
    temp_dir = tempfile.mkdtemp()
    try:
        real_dir = os.path.join(temp_dir, "real")
        fake_dir = os.path.join(temp_dir, "fake")
        os.makedirs(real_dir)
        os.makedirs(fake_dir)

        # Create 2 dummy real images and 3 dummy fake images
        for i in range(2):
            img = Image.new("RGB", (64, 64), color=(i * 10, i * 20, i * 30))
            img.save(os.path.join(real_dir, f"real_{i}.jpg"))

        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(255 - i * 10, 255 - i * 20, 255 - i * 30))
            img.save(os.path.join(fake_dir, f"fake_{i}.png"))

        dataset = BinaryImageDataset(root_dir=temp_dir)
        assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"

        counts = dataset.get_class_counts()
        assert counts["real"] == 2, f"Expected 2 real, got {counts['real']}"
        assert counts["fake"] == 3, f"Expected 3 fake, got {counts['fake']}"

        tensor, label, info = dataset[0]
        assert tensor.shape == (3, 224, 224)
        assert label.item() in (0.0, 1.0)
        assert "generator" in info
        print("test_binary_image_dataset passed.")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_clip_transform()
    test_fourier_spectrum()
    test_binary_image_dataset()
    print("All Stage 1 dataset tests passed successfully!")
