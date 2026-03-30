"""
Graph-Constrained Fusion Module for MedZFS (Section 4.4).

Constructs episode graph and performs GNN message passing:
    p_tilde = GNN(P_c^{(0)}, P_hard_c, H_tilde, F_q)    (Eq. 19)
"""

import torch
import torch.nn as nn


class GraphConstrainedFusion(nn.Module):
    """Fuse hallucinated and visual prototypes via graph-based attention."""

    def __init__(self, feature_dim=512, num_gnn_layers=2, attention_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.proto_anatomy_attention = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, attention_heads, dropout=dropout, batch_first=True)
            for _ in range(num_gnn_layers)
        ])
        self.proto_self_attention = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, attention_heads, dropout=dropout, batch_first=True)
            for _ in range(num_gnn_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, feature_dim * 4), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(feature_dim * 4, feature_dim), nn.Dropout(dropout))
            for _ in range(num_gnn_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_gnn_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_gnn_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_gnn_layers)])

    def forward(self, hallucinated_prototypes, hard_prototypes, graph_embeddings):
        """Fuse all prototypes through graph-constrained attention.

        Args:
            hallucinated_prototypes: (B, M_h, d)
            hard_prototypes: (B, M_v, d) or (B, 0, d) for zero-shot
            graph_embeddings: (B, N, d)

        Returns:
            Fused prototypes (B, M_fused, d), L2 normalized.
        """
        if hard_prototypes is not None and hard_prototypes.size(1) > 0:
            prototypes = torch.cat([hallucinated_prototypes, hard_prototypes], dim=1)
        else:
            prototypes = hallucinated_prototypes

        for i in range(len(self.proto_anatomy_attention)):
            attn_out, _ = self.proto_anatomy_attention[i](prototypes, graph_embeddings, graph_embeddings)
            prototypes = self.norm1[i](prototypes + attn_out)
            self_out, _ = self.proto_self_attention[i](prototypes, prototypes, prototypes)
            prototypes = self.norm2[i](prototypes + self_out)
            prototypes = self.norm3[i](prototypes + self.ffn_layers[i](prototypes))

        return nn.functional.normalize(prototypes, p=2, dim=-1)

    def compute_similarity_map(self, query_features, prototypes, temperature=0.1):
        """Compute cosine similarity map (Eq. 4/10).

        Args:
            query_features: (B, d, H, W)
            prototypes: (B, M, d)
            temperature: scaling factor

        Returns:
            Similarity map (B, H, W)
        """
        B, d, H, W = query_features.shape
        features_flat = query_features.view(B, d, H * W).permute(0, 2, 1)
        similarity = torch.bmm(features_flat, prototypes.permute(0, 2, 1)) / temperature
        max_sim, _ = similarity.max(dim=-1)
        return max_sim.view(B, H, W)
