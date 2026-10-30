"""Feature extraction and caching engine.

Precomputes and caches frozen CLIP visual embeddings phi(x) across dataset splits.
This enables instant linear probe training and nearest-neighbor evaluation
without repeated forward passes through the vision transformer backbone.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.detector import UniversalFakeDetector


def extract_and_cache_features(
    detector: UniversalFakeDetector,
    dataloader: DataLoader,
    output_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Extracts normalized embeddings from dataloader and caches them to disk.
    
    Args:
        detector: UniversalFakeDetector instance (with frozen backbone).
        dataloader: PyTorch DataLoader yielding (images, labels, metadata).
        output_path: File path to save the .pt feature cache.
        device: Device to run extraction on (CPU or CUDA).
        
    Returns:
        Dictionary containing extracted 'features', 'labels', and 'metadata'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector.to(device)
    detector.eval()

    all_features = []
    all_labels = []
    all_generators = []
    all_paths = []

    print(f"Extracting frozen CLIP representations on device: {device}...")
    with torch.no_grad():
        for images, labels, info in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = detector.extract_features(images, normalize=True)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            if isinstance(info, dict) and "generator" in info:
                all_generators.extend(info["generator"])
                all_paths.extend(info["path"])

    stacked_features = torch.cat(all_features, dim=0)
    stacked_labels = torch.cat(all_labels, dim=0)

    cache_data = {
        "features": stacked_features,
        "labels": stacked_labels,
        "generators": all_generators,
        "paths": all_paths,
        "feature_dim": stacked_features.shape[-1],
        "total_samples": stacked_features.shape[0],
        "model_name": detector.model_name,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(cache_data, output_path)
    print(f"Features successfully cached to: {output_path}")
    print(f"Feature matrix shape: {stacked_features.shape}, Labels shape: {stacked_labels.shape}")

    return cache_data


def load_cached_features(cache_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Loads a precomputed feature cache dictionary from disk."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached feature file not found at: {cache_path}")
    return torch.load(cache_path, map_location=map_location)
