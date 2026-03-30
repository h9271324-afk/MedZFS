"""
Hard Prototype Mining for MedZFS.

Implements the Few-Shot Hard Prototype Refinement module (Section 4.3)
that mines hard foreground prototypes from support set images where
the hallucination-based prediction fails.

Identifies hard regions:
    Ω_hf_i = {x | Y_i(x) = 1, Ŷ_i^{(0)}(x) = 0}    (Eq. 9)

Then computes hard prototypes via stochastic subsampling:
    p_hard,j_c = (1/|Ω_j|) Σ_{x∈Ω_j} F_i(x)          (Eq. 17)
"""

import torch
import torch.nn as nn


class HardPrototypeMiner(nn.Module):
    """Mine hard foreground prototypes from support set examples.

    During few-shot inference (k > 0), this module identifies pixels where
    the zero-shot hallucination prediction misses the true foreground.
    Features from these "hard" regions are aggregated into hard prototypes
    that complement the hallucinated prototypes.

    This ensures the model adapts to visual patterns not captured by
    text-only hallucination, such as unusual texture or boundary details.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_hard_prototypes: int = 8,
        hard_threshold: float = 0.5,
        subsample_ratio: float = 0.5,
    ):
        """Initialize the hard prototype miner.

        Args:
            feature_dim: Feature dimension (d).
            num_hard_prototypes: Number of hard prototypes to extract per
                                support image.
            hard_threshold: Prediction threshold for identifying hard pixels.
            subsample_ratio: Ratio of hard pixels to randomly subsample
                            for each prototype.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_hard_prototypes = num_hard_prototypes
        self.hard_threshold = hard_threshold
        self.subsample_ratio = subsample_ratio

        # Alignment projection: maps hard prototypes to be comparable
        # with hallucinated prototypes (Π in Eq. 18)
        self.alignment_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def identify_hard_regions(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        """Identify hard foreground pixels where hallucination fails.

        Hard foreground: pixels that are in the ground truth mask but
        are NOT predicted by the hallucination-based zero-shot model.

        Args:
            prediction: Zero-shot prediction of shape (B, H, W), values in [0, 1].
            ground_truth: Ground truth binary mask of shape (B, H, W).

        Returns:
            Hard foreground mask of shape (B, H, W), binary.
        """
        pred_binary = (prediction > self.hard_threshold).float()
        hard_mask = ground_truth * (1.0 - pred_binary)
        return hard_mask

    def mine_prototypes(
        self,
        features: torch.Tensor,
        hard_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hard prototypes from hard foreground regions.

        Stochastically subsamples hard pixels and computes mean-pooled
        prototypes.

        Args:
            features: Dense feature map (B, d, H, W) from the visual encoder.
            hard_mask: Hard foreground mask (B, H, W), binary.

        Returns:
            Hard prototypes of shape (B, num_hard_prototypes, d).
        """
        B, d, H, W = features.shape
        device = features.device

        # Reshape features to (B, d, H*W)
        features_flat = features.view(B, d, H * W)  # (B, d, N)
        hard_mask_flat = hard_mask.view(B, H * W)    # (B, N)

        hard_prototypes = []
        for b in range(B):
            # Find hard pixel indices for this sample
            hard_indices = torch.nonzero(hard_mask_flat[b] > 0.5, as_tuple=False).squeeze(-1)

            if hard_indices.numel() < 5:
                # Not enough hard pixels — use mean of all foreground
                fg_indices = torch.nonzero(hard_mask_flat[b] >= 0, as_tuple=False).squeeze(-1)
                if fg_indices.numel() > 0:
                    proto = features_flat[b, :, fg_indices].mean(dim=-1)  # (d,)
                else:
                    proto = torch.zeros(d, device=device)
                protos = proto.unsqueeze(0).expand(self.num_hard_prototypes, -1)
            else:
                protos = []
                for _ in range(self.num_hard_prototypes):
                    # Stochastic subsample of hard pixels
                    num_sample = max(1, int(hard_indices.size(0) * self.subsample_ratio))
                    perm = torch.randperm(hard_indices.size(0), device=device)[:num_sample]
                    subset = hard_indices[perm]

                    # Mean pool features from the subsample
                    proto = features_flat[b, :, subset].mean(dim=-1)  # (d,)
                    protos.append(proto)
                protos = torch.stack(protos, dim=0)  # (num_hard, d)

            hard_prototypes.append(protos)

        hard_prototypes = torch.stack(hard_prototypes, dim=0)  # (B, num_hard, d)

        # Align hard prototypes with hallucinated prototypes via projection
        hard_prototypes = self.alignment_projection(hard_prototypes)

        # L2 normalize for cosine similarity
        hard_prototypes = nn.functional.normalize(hard_prototypes, p=2, dim=-1)

        return hard_prototypes

    def forward(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
        zero_shot_predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Mine hard prototypes from the support set.

        Args:
            support_features: Support set dense features (B, k, d, H, W).
            support_masks: Support set ground truth masks (B, k, H, W).
            zero_shot_predictions: Zero-shot predictions on support (B, k, H, W).

        Returns:
            Hard prototypes of shape (B, k * num_hard_prototypes, d).
        """
        B, k, d, H, W = support_features.shape

        all_hard = []
        for i in range(k):
            # Identify hard regions for the i-th support example
            hard_mask = self.identify_hard_regions(
                zero_shot_predictions[:, i],
                support_masks[:, i],
            )  # (B, H, W)

            # Mine prototypes from hard regions
            hard_protos = self.mine_prototypes(
                support_features[:, i],
                hard_mask,
            )  # (B, num_hard, d)

            all_hard.append(hard_protos)

        # Concatenate across support examples
        hard_prototypes = torch.cat(all_hard, dim=1)  # (B, k * num_hard, d)

        return hard_prototypes
