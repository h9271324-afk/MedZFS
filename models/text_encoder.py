"""
Text Encoder for MedZFS.

Implements a BioClinicalBERT text encoder that maps anatomical
descriptions to d-dimensional embeddings in the shared feature space.
Frozen during early training stages to preserve pre-trained semantic
alignment, then unfrozen for fine-tuning in stage 3.

Maps: T_c → z_c ∈ R^d  (Eq. 2 in the paper)
"""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """BioClinicalBERT text encoder for anatomical descriptions.

    Encodes natural-language anatomical descriptions into dense vectors
    aligned with the visual feature space. Uses [CLS] token pooling
    followed by a learned projection head.

    Architecture:
        Text → BioClinicalBERT → [CLS] embedding → Projection → z_c ∈ R^d
    """

    def __init__(
        self,
        feature_dim: int = 512,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
        freeze: bool = True,
    ):
        """Initialize the text encoder.

        Args:
            feature_dim: Output embedding dimension (d).
            model_name: Hugging Face model identifier for the text backbone.
            max_length: Maximum input token length.
            freeze: Whether to freeze the backbone (True during stages 1-2).
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.max_length = max_length
        self.frozen = freeze

        # Load pre-trained BioClinicalBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Get the backbone's hidden size
        backbone_dim = self.backbone.config.hidden_size  # typically 768

        # Projection head: map from BERT hidden size to shared feature space
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Freeze backbone if specified
        if self.frozen:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all parameters in the BERT backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the BERT backbone for fine-tuning (stage 3)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.frozen = False

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text descriptions into embeddings.

        Args:
            texts: List of anatomical description strings.

        Returns:
            Text embeddings of shape (N, d), L2 normalized.
        """
        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to the same device as model parameters
        device = next(self.projection.parameters()).device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Forward through BERT
        if self.frozen:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (N, backbone_dim)

        # Project to shared feature space
        text_features = self.projection(cls_embedding)  # (N, d)

        # L2 normalize for cosine similarity
        text_features = nn.functional.normalize(text_features, p=2, dim=-1)

        return text_features

    def forward(
        self, texts: Optional[List[str]] = None, embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Either provide raw text strings or pre-computed token embeddings.

        Args:
            texts: List of anatomical description strings.
            embeddings: Pre-computed BERT embeddings (N, backbone_dim).

        Returns:
            Text features of shape (N, d), L2 normalized.
        """
        if texts is not None:
            return self.encode_text(texts)
        elif embeddings is not None:
            projected = self.projection(embeddings)
            return nn.functional.normalize(projected, p=2, dim=-1)
        else:
            raise ValueError("Either 'texts' or 'embeddings' must be provided.")
