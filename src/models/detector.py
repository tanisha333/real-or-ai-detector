"""Universal Fake Image Detector Architecture.

Implements the methodology of Ojha, Li, and Lee (CVPR 2023):
"Towards Universal Fake Image Detectors that Generalize Across Generative Models".

Architecture:
1. Backbone (phi): Frozen Vision-Language Foundation Model (CLIP ViT).
   All parameters are frozen (requires_grad = False).
2. Head (psi): Trainable Linear Probe (Linear layer + Sigmoid) or
   non-parametric Cosine Nearest-Neighbor feature bank.
"""

import os
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, AutoConfig


class UniversalFakeDetector(nn.Module):
    """Universal AI-Generated Image Detector using frozen CLIP vision representations."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        """Args:
            model_name: Hugging Face model identifier for the CLIP vision encoder.
                        Supports 'openai/clip-vit-base-patch32' (d=512) and
                        'openai/clip-vit-large-patch14' (d=768).
            pretrained: Whether to load official pretrained weights.
            dropout: Optional dropout rate before the linear layer (default 0.0 per paper).
        """
        super().__init__()
        self.model_name = model_name

        # 1. Load CLIP Vision Backbone
        if pretrained:
            try:
                # Try loading from local Hugging Face cache first for fast offline operation
                self.backbone = CLIPVisionModelWithProjection.from_pretrained(
                    model_name, local_files_only=True
                )
            except Exception:
                # Fallback to standard download if not already cached
                self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = CLIPVisionModelWithProjection(config)

        # 2. Freeze all parameters in the visual backbone (Gradient cutoff)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Dimension of the projected vision embedding (e.g., 512 for ViT-B/32, 768 for ViT-L/14)
        self.feature_dim = self.backbone.config.projection_dim

        # 3. Lightweight Classification Probe (psi)
        # In Ojha et al., a single linear layer maps the frozen feature to binary classification
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.linear_probe = nn.Linear(self.feature_dim, 1)

    def train(self, mode: bool = True):
        """Ensures that the CLIP backbone remains permanently in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extracts deep visual representation vectors phi(x) from the frozen CLIP backbone.
        
        Args:
            images: Tensor of shape (Batch_Size, 3, 224, 224), preprocessed to CLIP standard.
            normalize: Whether to apply L2 normalization (standard practice for CLIP embeddings).
            
        Returns:
            Feature tensor of shape (Batch_Size, feature_dim).
        """
        outputs = self.backbone(pixel_values=images)
        # image_embeds is the pooled, projected representation in vision-language space
        features = outputs.image_embeds
        if normalize:
            features = F.normalize(features, p=2, dim=-1)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for end-to-end inference or gradient updates on the probe.
        
        Args:
            images: Tensor of shape (Batch_Size, 3, 224, 224).
            
        Returns:
            Raw classification logits of shape (Batch_Size, 1).
        """
        with torch.no_grad():
            features = self.extract_features(images, normalize=True)
        
        features = self.dropout(features)
        logits = self.linear_probe(features)
        return logits

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass directly from pre-extracted/cached features.
        
        Enables lightning-fast training of the linear probe without recomputing
        forward passes through the massive vision transformer backbone.
        """
        features = self.dropout(features)
        logits = self.linear_probe(features)
        return logits

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Predicts the probability of an image being AI-generated: P(Fake | x).
        
        Returns:
            1D Tensor of probabilities in range [0.0, 1.0].
        """
        logits = self.forward(images)
        probabilities = torch.sigmoid(logits).squeeze(-1)
        return probabilities

    def save_probe(self, filepath: str):
        """Saves only the trainable linear probe parameters (lightweight, ~2-4 KB)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "linear_probe_state": self.linear_probe.state_dict(),
        }, filepath)

    def load_probe(self, filepath: str, map_location: str = "cpu"):
        """Loads linear probe weights into the detector."""
        checkpoint = torch.load(filepath, map_location=map_location)
        self.linear_probe.load_state_dict(checkpoint["linear_probe_state"])


class NearestNeighborClassifier:
    """Non-parametric Nearest Neighbor Classifier using frozen CLIP feature embeddings.
    
    Implements Section 4 (Figure 4) of Ojha et al. (CVPR 2023).
    Given training banks phi_R (real) and phi_F (fake), a test image x is classified
    by comparing its minimum cosine distance to phi_R vs phi_F:
        pred(x) = 1 if min_i d(phi_x, phi_fi) < min_i d(phi_x, phi_ri) else 0
    """

    def __init__(self):
        self.real_bank: Optional[torch.Tensor] = None  # (N_real, d)
        self.fake_bank: Optional[torch.Tensor] = None  # (N_fake, d)

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """Stores normalized training embeddings partitioned into real and fake banks."""
        features = F.normalize(features, p=2, dim=-1)
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        self.real_bank = features[real_mask].clone()
        self.fake_bank = features[fake_mask].clone()

    def predict(self, query_features: torch.Tensor) -> torch.Tensor:
        """Predicts binary class (0: real, 1: fake) based on minimum cosine distance.
        
        Args:
            query_features: Tensor of shape (Batch_Size, d).
            
        Returns:
            Predictions tensor of shape (Batch_Size,) with values in {0, 1}.
        """
        if self.real_bank is None or self.fake_bank is None:
            raise RuntimeError("Classifier has not been fitted with a feature bank yet.")

        query_norm = F.normalize(query_features, p=2, dim=-1)

        # Cosine distance = 1 - cosine_similarity
        # similarity = query @ bank.T
        real_sim = torch.matmul(query_norm, self.real_bank.T)  # (B, N_real)
        fake_sim = torch.matmul(query_norm, self.fake_bank.T)  # (B, N_fake)

        # Minimum distance corresponds to maximum similarity
        max_real_sim, _ = torch.max(real_sim, dim=-1)  # (B,)
        max_fake_sim, _ = torch.max(fake_sim, dim=-1)  # (B,)

        # If max_fake_sim > max_real_sim, then min_fake_dist < min_real_dist -> Fake (1)
        predictions = (max_fake_sim > max_real_sim).long()
        return predictions
