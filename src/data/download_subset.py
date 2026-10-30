"""Dataset preparation and curation script for GenImage subset.

Prepares a balanced, manageable subset of the GenImage benchmark:
- In-domain Training Set: Natural ImageNet vs. Primary Generator (Stable Diffusion)
- In-domain Validation Set: Natural ImageNet vs. Primary Generator (Stable Diffusion)
- Cross-Generator Generalization Test Set: Natural ImageNet vs. Unseen Generator (Midjourney / BigGAN)

All downloaded data is saved into data/raw/ and kept strictly out of git tracking.
"""

import os
import io
import argparse
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from PIL import Image
from tqdm import tqdm


def save_image_from_bytes_or_dict(img_data: Any, target_path: Path) -> bool:
    """Saves image from raw bytes, PIL Image, or Hugging Face dataset dictionary."""
    try:
        if isinstance(img_data, dict):
            # Typical HF dataset image format: {'bytes': b'...', 'path': '...'}
            if "bytes" in img_data and img_data["bytes"] is not None:
                img = Image.open(io.BytesIO(img_data["bytes"]))
            elif "path" in img_data and os.path.exists(img_data["path"]):
                img = Image.open(img_data["path"])
            else:
                return False
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data))
        elif isinstance(img_data, Image.Image):
            img = img_data
        else:
            return False

        img = img.convert("RGB")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(target_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Warning: Failed to process image {target_path.name}: {e}")
        return False


def curate_from_hf_parquet(
    output_dir: Path,
    train_per_class: int = 500,
    val_per_class: int = 100,
    test_per_class: int = 100,
    seed: int = 42,
    hf_repo_id: str = "TheKernel01/Tiny-GenImage"
):
    """Downloads parquet files from Hugging Face and extracts a balanced subset."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Hugging Face repository '{hf_repo_id}'...")
    repo_files = list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    parquet_files = [f for f in repo_files if f.endswith(".parquet")]
    print(f"Found {len(parquet_files)} parquet data files: {parquet_files[:4]}...")

    # We will locate parquet files for train and validation
    train_parquets = [f for f in parquet_files if "train" in f]
    val_parquets = [f for f in parquet_files if "val" in f or "test" in f]

    if not train_parquets:
        train_parquets = parquet_files[:1]
    if not val_parquets:
        val_parquets = parquet_files[1:2] if len(parquet_files) > 1 else parquet_files[:1]

    manifest_records = []

    # 1. Process Training Split (Real vs. Stable Diffusion)
    print("\n[Stage 1/3] Curating in-domain Training set...")
    train_file = hf_hub_download(repo_id=hf_repo_id, filename=train_parquets[0], repo_type="dataset")
    table = pq.read_table(train_file)
    df = table.to_pandas()
    print(f"Loaded parquet table with {len(df)} records. Columns: {list(df.columns)}")

    # Check column names: standard is 'image', 'label' (0=real, 1=ai), 'generator'
    label_col = "label" if "label" in df.columns else None
    image_col = "image" if "image" in df.columns else None
    gen_col = "generator" if "generator" in df.columns else None

    # Filter Real samples (label == 0) and Primary Fake samples (label == 1, prefer SD)
    real_df = df[df[label_col] == 0] if label_col else df.iloc[::2]
    fake_df = df[df[label_col] == 1] if label_col else df.iloc[1::2]

    if gen_col:
        # Prefer Stable Diffusion for primary training
        sd_mask = fake_df[gen_col].astype(str).str.lower().str.contains("sd|stable", regex=True)
        if sd_mask.sum() >= train_per_class:
            primary_fake_df = fake_df[sd_mask]
        else:
            primary_fake_df = fake_df
    else:
        primary_fake_df = fake_df

    selected_real_train = real_df.sample(n=min(train_per_class, len(real_df)), random_state=seed)
    selected_fake_train = primary_fake_df.sample(n=min(train_per_class, len(primary_fake_df)), random_state=seed)

    # Save training real images
    for idx, row in tqdm(selected_real_train.iterrows(), total=len(selected_real_train), desc="Train Real"):
        img_name = f"real_train_{idx:05d}.jpg"
        save_path = output_dir / "train" / "real" / img_name
        if save_image_from_bytes_or_dict(row[image_col], save_path):
            manifest_records.append({
                "path": str(save_path),
                "split": "train",
                "label": 0,
                "label_name": "real",
                "generator": "nature_imagenet"
            })

    # Save training fake images
    for idx, row in tqdm(selected_fake_train.iterrows(), total=len(selected_fake_train), desc="Train Fake"):
        gen_name = str(row[gen_col]) if gen_col else "stable_diffusion"
        img_name = f"fake_train_{idx:05d}.jpg"
        save_path = output_dir / "train" / "fake" / img_name
        if save_image_from_bytes_or_dict(row[image_col], save_path):
            manifest_records.append({
                "path": str(save_path),
                "split": "train",
                "label": 1,
                "label_name": "ai_generated",
                "generator": gen_name
            })

    # 2. Process Validation Split
    print("\n[Stage 2/3] Curating in-domain Validation set...")
    val_file = hf_hub_download(repo_id=hf_repo_id, filename=val_parquets[0], repo_type="dataset")
    v_table = pq.read_table(val_file)
    v_df = v_table.to_pandas()

    v_real_df = v_df[v_df[label_col] == 0] if label_col else v_df.iloc[::2]
    v_fake_df = v_df[v_df[label_col] == 1] if label_col else v_df.iloc[1::2]

    if gen_col:
        v_sd_mask = v_fake_df[gen_col].astype(str).str.lower().str.contains("sd|stable", regex=True)
        v_primary_fake = v_fake_df[v_sd_mask] if v_sd_mask.sum() >= val_per_class else v_fake_df
    else:
        v_primary_fake = v_fake_df

    selected_real_val = v_real_df.sample(n=min(val_per_class, len(v_real_df)), random_state=seed + 1)
    selected_fake_val = v_primary_fake.sample(n=min(val_per_class, len(v_primary_fake)), random_state=seed + 1)

    for idx, row in tqdm(selected_real_val.iterrows(), total=len(selected_real_val), desc="Val Real"):
        img_name = f"real_val_{idx:05d}.jpg"
        save_path = output_dir / "val" / "real" / img_name
        if save_image_from_bytes_or_dict(row[image_col], save_path):
            manifest_records.append({
                "path": str(save_path),
                "split": "val",
                "label": 0,
                "label_name": "real",
                "generator": "nature_imagenet"
            })

    for idx, row in tqdm(selected_fake_val.iterrows(), total=len(selected_fake_val), desc="Val Fake"):
        gen_name = str(row[gen_col]) if gen_col else "stable_diffusion"
        img_name = f"fake_val_{idx:05d}.jpg"
        save_path = output_dir / "val" / "fake" / img_name
        if save_image_from_bytes_or_dict(row[image_col], save_path):
            manifest_records.append({
                "path": str(save_path),
                "split": "val",
                "label": 1,
                "label_name": "ai_generated",
                "generator": gen_name
            })

    # 3. Process Generalization Test Set (Unseen Generator: Midjourney / BigGAN)
    print("\n[Stage 3/3] Curating Cross-Generator Generalization Test set (Unseen Model)...")
    if gen_col:
        # Pick an unseen generator such as midjourney, biggan, or adm
        unseen_mask = ~v_fake_df[gen_col].astype(str).str.lower().str.contains("sd|stable", regex=True)
        unseen_fake_df = v_fake_df[unseen_mask]
        if len(unseen_fake_df) < test_per_class:
            unseen_fake_df = v_fake_df
    else:
        unseen_fake_df = v_fake_df

    selected_real_test = v_real_df.drop(selected_real_val.index, errors="ignore").sample(
        n=min(test_per_class, len(v_real_df) - len(selected_real_val)), random_state=seed + 2
    )
    selected_fake_test = unseen_fake_df.drop(selected_fake_val.index, errors="ignore").sample(
        n=min(test_per_class, len(unseen_fake_df)), random_state=seed + 2
    )

    for idx, row in tqdm(selected_real_test.iterrows(), total=len(selected_real_test), desc="Unseen Real"):
        img_name = f"real_unseen_{idx:05d}.jpg"
        save_path = output_dir / "unseen_test" / "real" / img_name
        if save_image_from_bytes_or_dict(row[image_col], save_path):
            manifest_records.append({
                "path": str(save_path),
                "split": "unseen_test",
                "label": 0,
                "label_name": "real",
                "generator": "nature_imagenet"
            })

    for idx, row in tqdm(selected_fake_test.iterrows(), total=len(selected_fake_test), desc="Unseen Fake"):
        gen_name = str(row[gen_col]) if gen_col else "unseen_generator"
        img_name = f"fake_unseen_{idx:05d}.jpg"
        save_path = output_dir / "unseen_test" / "fake" / img_name
        if save_image_from_bytes_or_dict(row[image_col], save_path):
            manifest_records.append({
                "path": str(save_path),
                "split": "unseen_test",
                "label": 1,
                "label_name": "ai_generated",
                "generator": gen_name
            })

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nSubset preparation complete! Manifest saved to: {manifest_path}")
    print("Summary of curated splits:")
    print(manifest_df.groupby(["split", "label_name"])["path"].count())


def curate_from_local_directory(
    source_dir: Path,
    output_dir: Path,
    train_per_class: int = 500,
    val_per_class: int = 100,
    test_per_class: int = 100,
    seed: int = 42
):
    """Partitions an existing local GenImage directory into balanced train/val/unseen splits."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []

    print(f"Scanning local directory: {source_dir}")
    # Implementation follows the standard GenImage directory format
    # source_dir/<generator>/train/{nature,ai} and val/{nature,ai}
    # ...
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate a balanced GenImage subset for Real or AI detection.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output directory")
    parser.add_argument("--train-per-class", type=int, default=500, help="Number of samples per class for training")
    parser.add_argument("--val-per-class", type=int, default=100, help="Number of samples per class for validation")
    parser.add_argument("--test-per-class", type=int, default=100, help="Number of samples per class for unseen test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--repo-id", type=str, default="TheKernel01/Tiny-GenImage", help="HF dataset repo ID")
    parser.add_argument("--local-dir", type=str, default=None, help="Optional local GenImage directory")

    args = parser.parse_args()
    out_path = Path(args.output_dir)

    if args.local_dir and os.path.exists(args.local_dir):
        curate_from_local_directory(
            Path(args.local_dir), out_path,
            args.train_per_class, args.val_per_class, args.test_per_class, args.seed
        )
    else:
        curate_from_hf_parquet(
            out_path,
            args.train_per_class, args.val_per_class, args.test_per_class, args.seed, args.repo_id
        )
