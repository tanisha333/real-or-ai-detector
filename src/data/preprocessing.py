"""Image preprocessing and frequency-domain analysis pipeline.

Implements the standardization and high-frequency Fourier analysis
established by Ojha et al. (CVPR 2023) for universal fake image detection.
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch
from PIL import Image

# Standard OpenAI CLIP Vision Normalization Constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def get_clip_transform(target_size: int = 224):
    """Returns a callable transform mapping a PIL Image to a normalized PyTorch tensor.
    
    Adheres strictly to CLIP's standard evaluation protocol:
    1. Ensure RGB mode.
    2. Resize shorter edge to target_size using bicubic interpolation.
    3. Center crop to (target_size, target_size).
    4. Rescale pixel values to [0.0, 1.0].
    5. Standardize using CLIP pretraining mean and std.
    """
    def transform(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 1. Resize shorter side preserving aspect ratio
        w, h = image.size
        if w < h:
            new_w = target_size
            new_h = int(round(h * (target_size / w)))
        else:
            new_h = target_size
            new_w = int(round(w * (target_size / h)))
        
        image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        
        # 2. Center crop
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))
        
        # 3. Convert to float array [0, 1]
        arr = np.array(image, dtype=np.float32) / 255.0
        
        # 4. Standardize with CLIP mean and std
        arr = (arr - CLIP_MEAN) / CLIP_STD
        
        # 5. Transpose to PyTorch (Channels, Height, Width)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor

    return transform


def compute_fourier_spectrum(image: Union[Image.Image, np.ndarray], ksize: int = 3) -> np.ndarray:
    """Computes the 2D Fourier log-magnitude spectrum of high-frequency residuals.
    
    Following Section 3.2 and Figure 3 of Ojha et al. (CVPR 2023):
    1. Convert image to grayscale.
    2. Extract high-frequency components by subtracting median-blurred image.
    3. Perform 2D Fast Fourier Transform (FFT).
    4. Compute centered log-magnitude spectrum.
    
    Args:
        image: PIL Image or RGB numpy array.
        ksize: Kernel size for median blur (default 3).
        
    Returns:
        A 2D numpy array normalized to [0, 255] uint8 representing the frequency spectrum.
    """
    from scipy.ndimage import median_filter

    if isinstance(image, Image.Image):
        image = image.convert("L")
        img_arr = np.array(image, dtype=np.float32)
    else:
        if len(image.shape) == 3:
            # Grayscale conversion weights: 0.299 R + 0.587 G + 0.114 B
            img_arr = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        else:
            img_arr = image.astype(np.float32)

    # 1. High-pass residual: subtract median blurred signal
    blurred = median_filter(img_arr, size=ksize)
    residual = img_arr - blurred

    # 2. 2D Fast Fourier Transform
    f_transform = np.fft.fft2(residual)
    f_shift = np.fft.fftshift(f_transform)

    # 3. Log-magnitude spectrum
    magnitude_spectrum = np.log(1.0 + np.abs(f_shift))

    # 4. Normalize to [0, 255]
    min_val, max_val = magnitude_spectrum.min(), magnitude_spectrum.max()
    if max_val > min_val:
        norm_spectrum = ((magnitude_spectrum - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm_spectrum = np.zeros_like(magnitude_spectrum, dtype=np.uint8)

    return norm_spectrum
