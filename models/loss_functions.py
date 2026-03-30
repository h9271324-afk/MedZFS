"""
Loss Functions for MedZFS (Eq. 20).

L = L_seg + λ₁·L_graph + λ₂·L_align + λ₃·L_boundary

Components:
  - SegmentationLoss: BCE + Dice loss for mask prediction
  - GraphConsistencyLoss: enforces anatomical prototype relations (Eq. 16)
  - AlignmentLoss: cross-modal hallucinated-visual alignment (Eq. 18)
  - BoundaryLoss: distance-transform-based boundary refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    """Differentiable Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class SegmentationLoss(nn.Module):
    """Combined BCE + Dice loss for segmentation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined segmentation loss.

        Args:
            logits: Raw prediction logits (B, H, W).
            target: Ground truth binary mask (B, H, W).
        """
        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(torch.sigmoid(logits), target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class GraphConsistencyLoss(nn.Module):
    """Enforce anatomical relations between hallucinated prototypes (Eq. 16).

    L_graph = Σ_{(u,v,r)∈E} ||p_u - p_v - δ_r||²
    """

    def __init__(self, feature_dim: int = 512, num_relations: int = 4):
        super().__init__()
        # Learnable relation embedding δ_r for each relation type
        self.relation_embeddings = nn.Embedding(num_relations, feature_dim)

    def forward(
        self,
        prototypes: torch.Tensor,
        node_indices: torch.LongTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute graph consistency loss.

        Args:
            prototypes: Prototype embeddings (B, M, d) or node embeddings (N, d).
            node_indices: Indices mapping graph nodes to prototypes.
            edge_index: Edge indices (2, E).
            edge_type: Edge relation types (E,).
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=prototypes.device)

        if prototypes.dim() == 3:
            # Use first batch element
            prototypes = prototypes[0]

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        # Clamp indices to valid range
        max_idx = prototypes.size(0) - 1
        src_idx = src_idx.clamp(0, max_idx)
        tgt_idx = tgt_idx.clamp(0, max_idx)

        src_protos = prototypes[src_idx]
        tgt_protos = prototypes[tgt_idx]
        delta_r = self.relation_embeddings(edge_type)

        diff = src_protos - tgt_protos - delta_r
        return (diff ** 2).sum(dim=-1).mean()


class AlignmentLoss(nn.Module):
    """Cross-modal alignment loss between hallucinated and hard prototypes (Eq. 18).

    L_align = Σ_j ||p_hard,j - Π(p_hall,j)||²
    """

    def forward(
        self,
        hard_prototypes: torch.Tensor,
        hallucinated_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss.

        Args:
            hard_prototypes: (B, M_v, d) - already projected.
            hallucinated_prototypes: (B, M_h, d).
        """
        if hard_prototypes.size(1) == 0 or hallucinated_prototypes.size(1) == 0:
            return torch.tensor(0.0, device=hard_prototypes.device)

        # Match closest pairs via cosine similarity
        sim = torch.bmm(hard_prototypes, hallucinated_prototypes.permute(0, 2, 1))
        max_sim_idx = sim.argmax(dim=-1)  # (B, M_v)

        matched_hall = torch.gather(
            hallucinated_prototypes, 1,
            max_sim_idx.unsqueeze(-1).expand(-1, -1, hallucinated_prototypes.size(-1))
        )

        return F.mse_loss(hard_prototypes, matched_hall)


class BoundaryLoss(nn.Module):
    """Distance-transform-based boundary loss for precise boundary delineation."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss.

        Args:
            pred: Prediction probabilities (B, H, W).
            target: Ground truth binary mask (B, H, W).
        """
        # Compute distance transform on CPU
        target_np = target.detach().cpu().numpy()
        dist_maps = []
        for b in range(target_np.shape[0]):
            gt = target_np[b]
            if gt.sum() > 0 and (1 - gt).sum() > 0:
                pos_dist = distance_transform_edt(gt)
                neg_dist = distance_transform_edt(1 - gt)
                dist_map = neg_dist - pos_dist
                # Normalize
                dist_map = dist_map / (np.abs(dist_map).max() + 1e-8)
            else:
                dist_map = np.zeros_like(gt)
            dist_maps.append(dist_map)

        dist_maps = torch.tensor(
            np.stack(dist_maps), dtype=torch.float32, device=pred.device
        )

        return (pred * dist_maps).mean()


class MedZFSLoss(nn.Module):
    """Combined MedZFS loss function (Eq. 20).

    L = L_seg + λ₁·L_graph + λ₂·L_align + λ₃·L_boundary
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_relations: int = 4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        lambda_graph: float = 0.1,
        lambda_align: float = 0.05,
        lambda_boundary: float = 0.1,
    ):
        super().__init__()
        self.seg_loss = SegmentationLoss(bce_weight, dice_weight)
        self.graph_loss = GraphConsistencyLoss(feature_dim, num_relations)
        self.align_loss = AlignmentLoss()
        self.boundary_loss = BoundaryLoss()
        self.lambda_graph = lambda_graph
        self.lambda_align = lambda_align
        self.lambda_boundary = lambda_boundary

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        hallucinated_prototypes: torch.Tensor = None,
        hard_prototypes: torch.Tensor = None,
        graph_embeddings: torch.Tensor = None,
        edge_index: torch.LongTensor = None,
        edge_type: torch.LongTensor = None,
    ) -> dict:
        """Compute total loss.

        Returns:
            Dict with keys: total, seg, graph, align, boundary.
        """
        losses = {}

        # Segmentation loss
        losses["seg"] = self.seg_loss(logits, target)

        # Graph consistency loss
        if graph_embeddings is not None and edge_index is not None and edge_type is not None:
            losses["graph"] = self.graph_loss(
                graph_embeddings, None, edge_index, edge_type
            )
        else:
            losses["graph"] = torch.tensor(0.0, device=logits.device)

        # Alignment loss
        if hard_prototypes is not None and hallucinated_prototypes is not None:
            losses["align"] = self.align_loss(hard_prototypes, hallucinated_prototypes)
        else:
            losses["align"] = torch.tensor(0.0, device=logits.device)

        # Boundary loss
        pred_prob = torch.sigmoid(logits)
        losses["boundary"] = self.boundary_loss(pred_prob, target)

        # Total
        losses["total"] = (
            losses["seg"]
            + self.lambda_graph * losses["graph"]
            + self.lambda_align * losses["align"]
            + self.lambda_boundary * losses["boundary"]
        )

        return losses
