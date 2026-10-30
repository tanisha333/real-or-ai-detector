# Dataset Directory: GenImage Subset

This directory contains the curated subsets of the **GenImage** benchmark used for training and evaluating the Universal Fake Image Detector.

## Directory Structure (Git-Ignored)

```text
data/
├── raw/
│   ├── train/
│   │   ├── real/      (Natural ImageNet camera images, Label: 0)
│   │   └── fake/      (AI-generated images from primary generator, Label: 1)
│   ├── val/
│   │   ├── real/      (Held-out natural ImageNet images, Label: 0)
│   │   └── fake/      (Held-out AI-generated images from primary generator, Label: 1)
│   └── unseen_test/   (Cross-Generator Generalization Test Set)
│       ├── real/      (Natural images, Label: 0)
│       └── fake/      (AI-generated from an unseen architecture, e.g. Midjourney, Label: 1)
└── processed/
    ├── train_features.pt   (Cached frozen CLIP embeddings)
    ├── val_features.pt
    └── test_features.pt
```

## Dataset Origin & Balance
- **Source:** GenImage Benchmark (*Zhu et al., NeurIPS 2023*) / Tiny-GenImage curation.
- **Labels:**
  - `0` = Real Image (captured naturally by camera)
  - `1` = AI-Generated Image (synthesized by a generative model)
- **Primary Generator for In-Domain Training:** Stable Diffusion v1.4 (Diffusion model).
- **Unseen Generator for Generalization Benchmark:** Midjourney / BigGAN.
- **Strict Data Hygiene:** Raw images and extracted tensor files are strictly excluded from git tracking via `.gitignore`.
