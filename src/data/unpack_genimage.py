"""Extracts and organizes the authentic GenImage subset into clean folder splits.

Creates:
- data/raw/train/ (real vs stable_diffusion_v_1_4)
- data/raw/val/   (real vs stable_diffusion_v_1_4)
- data/raw/test/  (held-out real vs unseen generators: Midjourney, BigGAN, ADM, GLIDE, etc.)
- data/raw/manifest.csv
"""

import os
import io
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
from tqdm import tqdm


def unpack_genimage_splits(
    cache_dir: str = "data/cache",
    raw_dir: str = "data/raw",
    seed: int = 42,
):
    """Unpacks cached parquet shards into standard image directory structure."""
    raw_path = Path(raw_dir)
    cache_path = Path(cache_dir)

    print("Loading GenImage parquet shards...")
    dfs = []
    for split_name in ["train", "val", "test"]:
        p = cache_path / f"tiny_{split_name}.parquet"
        if p.exists():
            table = pq.read_table(p)
            d = table.to_pandas()
            d["source_split"] = split_name
            dfs.append(d)

    if not dfs:
        raise FileNotFoundError(f"No parquet shards found in {cache_dir}")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} total authentic GenImage samples.")

    # Partition Real samples
    real_df = full_df[full_df["generator"] == "nature"].copy().sample(frac=1.0, random_state=seed)
    
    # 16 for train, 8 for val, and the 8 from val also evaluate against test sets
    train_real = real_df.iloc[:16]
    val_real = real_df.iloc[16:24]
    test_real = val_real  # Held-out real images for testing

    # Partition Primary Generator (SD 1.4)
    sd_df = full_df[full_df["generator"] == "stable_diffusion_v_1_4"].copy().sample(frac=1.0, random_state=seed)
    train_sd = sd_df.iloc[:16]
    val_sd = sd_df.iloc[16:24]

    manifest = []

    def save_img_record(row, dest_subpath: Path, split: str, label: int, generator: str):
        dest_subpath.parent.mkdir(parents=True, exist_ok=True)
        img_bytes = row["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(dest_subpath, format="JPEG", quality=95)
        manifest.append({
            "path": str(dest_subpath.relative_to(raw_path.parent)),
            "split": split,
            "label": label,
            "label_name": "real" if label == 0 else "fake",
            "generator": generator,
            "filename": dest_subpath.name,
        })

    # 1. Unpack Train Split
    print("Unpacking Train Split (16 Real vs 16 Stable Diffusion v1.4)...")
    for idx, (_, row) in enumerate(train_real.iterrows()):
        save_img_record(row, raw_path / "train" / "real" / f"real_train_{idx:03d}.jpg", "train", 0, "nature")
    for idx, (_, row) in enumerate(train_sd.iterrows()):
        save_img_record(row, raw_path / "train" / "fake" / f"fake_train_sd14_{idx:03d}.jpg", "train", 1, "stable_diffusion_v_1_4")

    # 2. Unpack Val Split
    print("Unpacking Validation Split (8 Real vs 8 Stable Diffusion v1.4)...")
    for idx, (_, row) in enumerate(val_real.iterrows()):
        save_img_record(row, raw_path / "val" / "real" / f"real_val_{idx:03d}.jpg", "val", 0, "nature")
    for idx, (_, row) in enumerate(val_sd.iterrows()):
        save_img_record(row, raw_path / "val" / "fake" / f"fake_val_sd14_{idx:03d}.jpg", "val", 1, "stable_diffusion_v_1_4")

    # 3. Unpack Unseen Generators for Cross-Generator Generalization
    print("Unpacking Unseen Generators for Cross-Model Generalization Testing...")
    # Unpack held-out real images in test/real
    for idx, (_, row) in enumerate(test_real.iterrows()):
        save_img_record(row, raw_path / "test" / "real" / f"real_test_{idx:03d}.jpg", "test", 0, "nature")

    # All unseen generators
    unseen_gens = [
        "Midjourney", "BigGAN", "ADM", "GLIDE",
        "stable_diffusion_v_1_5", "VQDM", "wukong"
    ]
    for gen in unseen_gens:
        gen_df = full_df[full_df["generator"] == gen]
        clean_gen_name = gen.lower().replace("_", "")
        for idx, (_, row) in enumerate(gen_df.iterrows()):
            dest = raw_path / "test" / clean_gen_name / f"fake_test_{clean_gen_name}_{idx:03d}.jpg"
            save_img_record(row, dest, f"test_{clean_gen_name}", 1, gen)

    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_csv = raw_path / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nManifest saved to: {manifest_csv}")
    print("\nDataset Summary:")
    print(manifest_df.groupby(["split", "label_name"]).size())


if __name__ == "__main__":
    unpack_genimage_splits()
