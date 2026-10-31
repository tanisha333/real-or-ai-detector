"""Visualization utilities for diagnostic figures, metrics, and frequency spectra."""

from typing import Union, Optional, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
from src.data.preprocessing import compute_fourier_spectrum


def plot_confusion_matrix(
    cm: List[List[int]],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    class_names: List[str] = ["Real", "AI-Generated"]
) -> plt.Figure:
    """Renders a formatted confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)

    # Plot colored heatmap
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    # Print numbers inside cells
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix plot to: {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Receiver Operating Characteristic (ROC) Curve"
) -> plt.Figure:
    """Plots ROC Curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ROC plot to: {save_path}")

    return fig


def plot_frequency_spectrum_analysis(
    real_img: Image.Image,
    fake_img: Image.Image,
    generator_name: str = "AI Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots side-by-side comparison of real and fake images and their Fourier frequency spectra."""
    real_spec = compute_fourier_spectrum(real_img)
    fake_spec = compute_fourier_spectrum(fake_img)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(real_img)
    axes[0, 0].set_title("Real Image (Natural Camera)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(real_spec, cmap="inferno")
    axes[0, 1].set_title("Real Frequency Residual Spectrum")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(fake_img)
    axes[1, 0].set_title(f"Synthetic Image ({generator_name})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(fake_spec, cmap="inferno")
    axes[1, 1].set_title(f"{generator_name} Frequency Spectrum")
    axes[1, 1].axis("off")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved frequency spectrum comparison to: {save_path}")

    return fig
