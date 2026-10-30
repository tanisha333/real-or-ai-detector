# Preprocessing Pipeline Documentation

This document explains the image preprocessing methodology implemented for **"Real or AI? — Deep Learning-Based AI-Generated Image Detection"**, following the principles of Ojha et al. (*CVPR 2023*).

---

## 1. Why Preprocessing Matters in AI Image Detection

Unlike general object recognition where semantic meaning is invariant to resolution, deepfake / AI-generated image detection operates on both:
1. **Semantic consistency** (captured by CLIP's vision transformer embeddings).
2. **Subtle pixel-level artifacts** (e.g., generator-induced grid artifacts, blending inconsistencies, frequency abnormalities).

Applying aggressive data augmentation (such as severe random cropping, heavy Gaussian blurring, or aggressive JPEG compression during training) can artificially destroy the delicate generation fingerprints that differentiate real from synthetic images. Therefore, following Ojha et al. (CVPR 2023), our preprocessing relies on **deterministic, high-fidelity standardization**.

---

## 2. The CLIP ViT Preprocessing Pipeline

The primary vision backbone is OpenAI's **CLIP ViT** (`clip-vit-base-patch32` or `clip-vit-large-patch14`). Images passing into the frozen feature extractor $\phi$ undergo the standardized CLIP vision transformation:

```text
Input Image (RGB, arbitrary dimensions H x W x 3)
   │
   ▼
1. Resize (Bicubic Interpolation)
   - Shorter edge resized to 224 pixels (or 336 for ViT-L/14@336)
   - Aspect ratio preserved to avoid synthetic geometric warping
   │
   ▼
2. Center Crop (224 x 224 pixels)
   - Focuses on the core content region
   │
   ▼
3. Normalization to [0.0, 1.0] Tensor
   - Rescale pixel values: x / 255.0
   │
   ▼
4. Z-Score Standardization (CLIP Pretraining Statistics)
   - Mean:   [0.48145466, 0.4578275, 0.40821073]
   - StdDev: [0.26862954, 0.26130258, 0.27577711]
   │
   ▼
Output Tensor: (Batch_Size, 3, 224, 224)
```

### PyTorch Implementation Formula:
$$\hat{x}_{c, i, j} = \frac{\frac{x_{c, i, j}}{255.0} - \mu_c}{\sigma_c}$$
Where $c \in \{R, G, B\}$.

---

## 3. Frequency-Domain Preprocessing (Fourier Analysis)

In Section 3.2 and Figure 3 of the CVPR 2023 paper, Ojha et al. analyzed the high-frequency residual spectra of real and generated images. We implement this preprocessing as a diagnostic tool for both the research pipeline and the Streamlit UI:

1. **High-Pass Residual Filtering:**
   Given an image $I$, compute its median blurred version $I_{\text{blur}}$ using a $3 \times 3$ kernel:
   $$R = I - I_{\text{blur}}$$
2. **2D Fast Fourier Transform (FFT):**
   Compute the 2D discrete Fourier transform on the residual:
   $$\mathcal{F}(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} R(x, y) e^{-j 2\pi (\frac{ux}{M} + \frac{vy}{N})}$$
3. **Log-Magnitude Spectrum Centering:**
   Shift zero frequency components to the center and compute the log magnitude:
   $$S(u, v) = \log(1 + |\text{fftshift}(\mathcal{F}(u, v))|)$$

This spectrum reveals whether the image exhibits distinct periodic grid spikes (common in GAN upsampling layers) or smooth, diffuse noise characteristics (typical of diffusion models and authentic camera sensors).

---

## 4. Verification & Validation Rules
- **RGB Mode Enforcement:** All images (including single-channel grayscale or 4-channel RGBA) are converted to 3-channel RGB.
- **Corrupt Image Handling:** Images with truncated byte streams or non-image payloads are logged, safely discarded, and excluded from the dataset split.
- **Label Integrity:**
  - Real = `0`
  - AI-Generated = `1`
