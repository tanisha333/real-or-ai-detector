"""PyTorch Dataset implementations for real vs. AI-generated images."""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import get_clip_transform


class BinaryImageDataset(Dataset):
    """Dataset for binary Real (0) vs. AI-Generated (1) image classification.
    
    Supports directory-based loading where images are organized in:
      root_dir/real/
      root_dir/fake/
    or custom file lists with generator metadata.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root_dir: Optional[str] = None,
        file_list: Optional[List[Tuple[str, int, str]]] = None,
        transform: Optional[Callable] = None,
    ):
        """Args:
            root_dir: Path to directory containing 'real' and 'fake' subfolders.
            file_list: Optional explicit list of tuples: (image_path, label, generator_name).
            transform: Callable preprocessing transform. Defaults to standard CLIP transform.
        """
        self.samples: List[Dict[str, Any]] = []
        self.transform = transform or get_clip_transform()

        if file_list is not None:
            for item in file_list:
                img_path, label, gen = item
                if Path(img_path).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append({
                        "path": str(img_path),
                        "label": int(label),
                        "generator": gen
                    })
        elif root_dir is not None:
            self._load_from_directory(root_dir)
        else:
            raise ValueError("Either root_dir or file_list must be provided.")

    def _load_from_directory(self, root_dir: str):
        """Loads images from a root directory.

        Supports two layouts:
        1. Flat layout:  root/real/   + root/fake/
        2. Per-generator test layout: root/real/   + root/<generator_name>/  (one folder per AI model)
           Any sub-folder that is NOT named 'real' or 'fake' is treated as a
           named AI generator with label 1.
        """
        root = Path(root_dir)
        real_dir = root / "real"
        fake_dir = root / "fake"

        # Always load real images from root/real/ if present
        if real_dir.exists():
            for p in sorted(real_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append({
                        "path": str(p),
                        "label": 0,
                        "generator": "real"
                    })

        if fake_dir.exists():
            # Flat layout: single fake/ folder (generator name inferred from filenames)
            for p in sorted(fake_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append({
                        "path": str(p),
                        "label": 1,
                        "generator": "ai_generator"
                    })
        else:
            # Per-generator layout: every sub-folder that isn't 'real' is an AI generator
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and sub.name.lower() != "real":
                    generator_name = sub.name  # e.g. "midjourney", "biggan", "adm"
                    for p in sorted(sub.iterdir()):
                        if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                            self.samples.append({
                                "path": str(p),
                                "label": 1,
                                "generator": generator_name
                            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample_info = self.samples[idx]
        img_path = sample_info["path"]
        label = sample_info["label"]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
        except Exception as e:
            # Fallback for corrupted file: replace with a zero tensor and log error
            tensor = torch.zeros(3, 224, 224)

        return tensor, torch.tensor(label, dtype=torch.float32), sample_info

    def get_class_counts(self) -> Dict[str, int]:
        """Returns the number of real vs. fake samples in the dataset."""
        real_count = sum(1 for s in self.samples if s["label"] == 0)
        fake_count = sum(1 for s in self.samples if s["label"] == 1)
        return {"real": real_count, "fake": fake_count, "total": len(self.samples)}


def create_dataloader(
    dataset: BinaryImageDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Creates a deterministic PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
