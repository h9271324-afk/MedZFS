"""
MedZFS: Full Model Architecture.

Assembles all sub-modules into the unified MedZFS framework for
zero-to-few-shot medical image segmentation.

Three pathways:
  1. Hallucination: text + graph → P_c^{(0)}
  2. Refinement: support + hard mining → P_hard_c (k > 0 only)
  3. Fusion: integrate prototypes → segmentation Ŷ_q
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.visual_encoder import VisualEncoder
from models.text_encoder import TextEncoder
from models.hallucination import PrototypeHallucinator
from models.graph_network import HeterogeneousGraphNetwork
from models.fusion import GraphConstrainedFusion
from models.prototype_mining import HardPrototypeMiner


class MedZFS(nn.Module):
    """MedZFS: Unified Zero-to-Few-Shot Medical Image Segmentation.

    Dynamically adapts computation based on the number of support examples k:
      - k=0: Only hallucination + graph + fusion (zero-shot)
      - k>0: Hallucination + mining + graph + fusion (few-shot)
    """

    def __init__(self, config: dict):
        """Initialize MedZFS from configuration dictionary.

        Args:
            config: Model configuration with keys for each sub-module.
        """
        super().__init__()

        model_cfg = config.get("model", config)
        self.feature_dim = model_cfg.get("feature_dim", 512)

        # --- Visual Encoder ---
        ve_cfg = model_cfg.get("visual_encoder", {})
        self.visual_encoder = VisualEncoder(
            feature_dim=self.feature_dim,
            pretrained=ve_cfg.get("pretrained", "imagenet"),
            feature_blocks=ve_cfg.get("feature_blocks", [2, 3, 4]),
            freeze_bn=ve_cfg.get("freeze_bn", True),
        )

        # --- Text Encoder ---
        te_cfg = model_cfg.get("text_encoder", {})
        self.text_encoder = TextEncoder(
            feature_dim=self.feature_dim,
            model_name=te_cfg.get("model_name", "emilyalsentzer/Bio_ClinicalBERT"),
            max_length=te_cfg.get("max_length", 128),
            freeze=te_cfg.get("freeze", True),
        )

        # --- Anatomical Graph Network ---
        gn_cfg = model_cfg.get("graph_network", {})
        self.graph_network = HeterogeneousGraphNetwork(
            feature_dim=self.feature_dim,
            hidden_dim=gn_cfg.get("hidden_dim", 512),
            num_layers=gn_cfg.get("num_layers", 4),
            num_relation_types=gn_cfg.get("num_relation_types", 4),
            dropout=gn_cfg.get("dropout", 0.1),
            residual=gn_cfg.get("residual", True),
        )

        # --- Prototype Hallucination ---
        hall_cfg = model_cfg.get("hallucination", {})
        self.hallucinator = PrototypeHallucinator(
            feature_dim=self.feature_dim,
            num_samples=hall_cfg.get("num_samples", 16),
            min_variance=hall_cfg.get("min_variance", 0.01),
            max_variance=hall_cfg.get("max_variance", 2.0),
        )
        self.temperature = hall_cfg.get("temperature", 0.1)

        # --- Hard Prototype Mining ---
        pm_cfg = model_cfg.get("prototype_mining", {})
        self.prototype_miner = HardPrototypeMiner(
            feature_dim=self.feature_dim,
            num_hard_prototypes=pm_cfg.get("num_hard_prototypes", 8),
            hard_threshold=pm_cfg.get("hard_threshold", 0.5),
            subsample_ratio=pm_cfg.get("subsample_ratio", 0.5),
        )

        # --- Graph-Constrained Fusion ---
        fuse_cfg = model_cfg.get("fusion", {})
        self.fusion = GraphConstrainedFusion(
            feature_dim=self.feature_dim,
            num_gnn_layers=fuse_cfg.get("num_gnn_layers", 2),
            attention_heads=fuse_cfg.get("attention_heads", 8),
            dropout=fuse_cfg.get("dropout", 0.1),
        )

    def encode_anatomy(
        self,
        node_descriptions: List[str],
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode the anatomical graph into feature space.

        Args:
            node_descriptions: Text descriptions of each graph node.
            edge_index: (2, E) edge indices.
            edge_type: (E,) edge relation types.

        Returns:
            Graph node embeddings (N, d).
        """
        # Initialize node features from text encoder
        node_features = self.text_encoder(texts=node_descriptions)  # (N, d)

        # Run heterogeneous graph convolution
        graph_embeddings = self.graph_network(node_features, edge_index, edge_type)

        return graph_embeddings

    def zero_shot_predict(
        self,
        query_features: torch.Tensor,
        hallucinated_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute zero-shot prediction using hallucinated prototypes.

        Args:
            query_features: (B, d, H, W).
            hallucinated_prototypes: (B, M, d).

        Returns:
            Similarity map (B, H, W) before sigmoid.
        """
        return self.fusion.compute_similarity_map(
            query_features, hallucinated_prototypes, self.temperature
        )

    def forward(
        self,
        query_image: torch.Tensor,
        class_description: str,
        node_descriptions: List[str],
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        support_images: Optional[torch.Tensor] = None,
        support_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for zero-to-few-shot segmentation.

        Args:
            query_image: (B, 3, H, W).
            class_description: Text description of the target class.
            node_descriptions: Text descriptions for all graph nodes.
            edge_index: (2, E) graph edge indices.
            edge_type: (E,) edge relation types.
            support_images: (B, k, 3, H, W) or None for zero-shot.
            support_masks: (B, k, H, W) or None for zero-shot.

        Returns:
            Dict with: logits, prediction, hallucinated_prototypes,
            hard_prototypes, fused_prototypes, graph_embeddings.
        """
        B = query_image.size(0)
        device = query_image.device

        # Determine shot setting
        k = 0
        if support_images is not None and support_images.dim() == 5:
            k = support_images.size(1)

        # --- Pathway 1: Hallucination ---
        # 1a. Encode anatomical graph
        graph_embeddings = self.encode_anatomy(
            node_descriptions, edge_index, edge_type
        )  # (N, d)
        graph_embeddings_batch = graph_embeddings.unsqueeze(0).expand(B, -1, -1)

        # 1b. Encode target class text
        text_embedding = self.text_encoder(texts=[class_description])  # (1, d)
        text_embedding = text_embedding.expand(B, -1)  # (B, d)

        # 1c. Hallucinate prototypes
        hall_output = self.hallucinator(text_embedding, graph_embeddings_batch)
        hallucinated_prototypes = hall_output["prototypes"]  # (B, M, d)

        # --- Encode Query Image ---
        query_features = self.visual_encoder(query_image)  # (B, d, h, w)

        # --- Pathway 2: Refinement (k > 0 only) ---
        hard_prototypes = None
        if k > 0:
            # Encode support images
            support_flat = support_images.view(B * k, *support_images.shape[2:])
            support_features_flat = self.visual_encoder(support_flat)
            sfh, sfw = support_features_flat.shape[2], support_features_flat.shape[3]
            support_features = support_features_flat.view(
                B, k, self.feature_dim, sfh, sfw
            )

            # Compute zero-shot predictions on support set
            with torch.no_grad():
                zs_preds = []
                for i in range(k):
                    sf = support_features[:, i]  # (B, d, h, w)
                    zs_pred = self.zero_shot_predict(sf, hallucinated_prototypes)
                    zs_preds.append(torch.sigmoid(zs_pred))
                zero_shot_preds = torch.stack(zs_preds, dim=1)  # (B, k, h, w)

            # Resize support masks to feature map resolution
            sm_resized = nn.functional.interpolate(
                support_masks.unsqueeze(2).float(),
                size=(sfh, sfw), mode="nearest",
            ).squeeze(2)  # (B, k, h, w)

            # Mine hard prototypes
            hard_prototypes = self.prototype_miner(
                support_features, sm_resized, zero_shot_preds
            )  # (B, k*num_hard, d)

        # --- Pathway 3: Graph-Constrained Fusion ---
        fused_prototypes = self.fusion(
            hallucinated_prototypes,
            hard_prototypes if hard_prototypes is not None else torch.zeros(B, 0, self.feature_dim, device=device),
            graph_embeddings_batch,
        )  # (B, M_fused, d)

        # --- Segmentation ---
        similarity_map = self.fusion.compute_similarity_map(
            query_features, fused_prototypes, self.temperature
        )  # (B, h, w)

        # Upsample to original resolution
        logits = nn.functional.interpolate(
            similarity_map.unsqueeze(1),
            size=query_image.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        prediction = torch.sigmoid(logits)

        return {
            "logits": logits,
            "prediction": prediction,
            "hallucinated_prototypes": hallucinated_prototypes,
            "hard_prototypes": hard_prototypes,
            "fused_prototypes": fused_prototypes,
            "graph_embeddings": graph_embeddings,
            "mu": hall_output["mu"],
            "log_var": hall_output["log_var"],
        }
