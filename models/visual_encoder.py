"""
Visual Encoder for MedZFS.

Implements a ResNet-101 backbone with multi-scale feature extraction
from blocks 2, 3, and 4. Initialized from BiomedCLIP weights for
cross-modal alignment with the text encoder.

The encoder produces dense feature maps F_q ∈ R^{h×w×d} where d is
the shared feature dimension (default 512).
"""

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models


class VisualEncoder(nn.Module):
    """ResNet-101 visual encoder with multi-scale feature extraction.

    Extracts features from intermediate blocks of ResNet-101 and fuses
    them into a single dense feature map aligned with the text embedding
    space via a learned projection head.

    Architecture:
        Input (3, H, W) → ResNet-101 blocks → Multi-scale features
        → FPN-style fusion → Projection → Dense feature map (d, h, w)
    """

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: str = "biomedclip",
        feature_blocks: List[int] = None,
        freeze_bn: bool = True,
    ):
        """Initialize the visual encoder.

        Args:
            feature_dim: Output feature dimension (d).
            pretrained: Pretrained weight source. "biomedclip" for
                       BiomedCLIP initialization, "imagenet" for ImageNet,
                       or "none" for random initialization.
            feature_blocks: Which ResNet blocks to extract features from.
                           Default: [2, 3, 4] for multi-scale context.
            freeze_bn: Whether to freeze BatchNorm layers.
        """
        super().__init__()

        if feature_blocks is None:
            feature_blocks = [2, 3, 4]

        self.feature_dim = feature_dim
        self.feature_blocks = feature_blocks
        self.freeze_bn = freeze_bn

        # Initialize ResNet-101 backbone
        if pretrained == "imagenet":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        else:
            backbone = models.resnet101(weights=None)

        # Extract backbone layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # Block 1: stride 4, 256 channels
        self.layer2 = backbone.layer2  # Block 2: stride 8, 512 channels
        self.layer3 = backbone.layer3  # Block 3: stride 16, 1024 channels
        self.layer4 = backbone.layer4  # Block 4: stride 32, 2048 channels

        # Channel dimensions for each block
        block_channels = {1: 256, 2: 512, 3: 1024, 4: 2048}

        # Lateral connections for FPN-style multi-scale fusion
        self.lateral_convs = nn.ModuleDict()
        self.output_convs = nn.ModuleDict()
        for block_idx in self.feature_blocks:
            in_ch = block_channels[block_idx]
            self.lateral_convs[str(block_idx)] = nn.Conv2d(
                in_ch, feature_dim, kernel_size=1, bias=False
            )
            self.output_convs[str(block_idx)] = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
            )

        # Final projection to shared feature space
        self.projection = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

        # Freeze BatchNorm layers if specified
        if self.freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self) -> None:
        """Freeze all BatchNorm layers in the backbone."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train mode to keep BN frozen."""
        super().train(mode)
        if self.freeze_bn:
            self._freeze_bn()
        return self

    def _extract_block_features(self, x: torch.Tensor) -> dict:
        """Extract features from each ResNet block.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Dictionary mapping block index to feature tensor.
        """
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 1 in self.feature_blocks:
            features[1] = x

        x = self.layer2(x)
        if 2 in self.feature_blocks:
            features[2] = x

        x = self.layer3(x)
        if 3 in self.feature_blocks:
            features[3] = x

        x = self.layer4(x)
        if 4 in self.feature_blocks:
            features[4] = x

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dense visual feature map.

        Extracts multi-scale features from the ResNet backbone and fuses
        them via FPN-style top-down pathway with lateral connections.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dense feature map of shape (B, d, h, w) where h, w depend on
            the stride of the highest-resolution extracted block.
        """
        block_features = self._extract_block_features(x)

        # Apply lateral 1x1 convolutions
        lateral_features = {}
        for block_idx in sorted(self.feature_blocks, reverse=True):
            feat = block_features[block_idx]
            lateral_features[block_idx] = self.lateral_convs[str(block_idx)](feat)

        # Top-down pathway: coarsest to finest
        sorted_blocks = sorted(self.feature_blocks, reverse=True)
        fused = lateral_features[sorted_blocks[0]]

        for i in range(1, len(sorted_blocks)):
            block_idx = sorted_blocks[i]
            target_size = lateral_features[block_idx].shape[2:]
            upsampled = nn.functional.interpolate(
                fused, size=target_size, mode="bilinear", align_corners=False
            )
            fused = lateral_features[block_idx] + upsampled

        # Apply output convolution
        fused = self.output_convs[str(sorted_blocks[-1])](fused)

        # Project to shared feature space
        features = self.projection(fused)

        # L2 normalize along the channel dimension for cosine similarity
        features = nn.functional.normalize(features, p=2, dim=1)

        return features
