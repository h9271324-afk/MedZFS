"""
Heterogeneous Graph Neural Network for MedZFS.

Implements the anatomical graph embedding module (Section 4.1) using
heterogeneous graph convolution with L layers and learnable edge-type-specific
weight matrices W_r.

Computes:
    h_v^{(l+1)} = Σ_{r∈R} Σ_{u∈N_r(v)} W_r^{(l)} · h_u^{(l)}    (Eq. 13)

Supports relation types: spatial, hierarchical, functional, pathological.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class HeterogeneousGraphConvLayer(nn.Module):
    """Single layer of heterogeneous graph convolution.

    Performs message passing with relation-specific weight matrices.
    Each edge type has its own learned linear transformation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relation_types: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        """Initialize the graph convolution layer.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            num_relation_types: Number of edge relation types R.
            dropout: Dropout probability.
            residual: Whether to use residual connections.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relation_types = num_relation_types
        self.residual = residual

        # Relation-specific weight matrices W_r
        self.relation_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relation_types)
        ])

        # Self-loop transformation
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual projection if dimensions differ
        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.Tensor:
        """Perform one round of heterogeneous message passing.

        Args:
            node_features: Node feature matrix (N, in_dim).
            edge_index: Edge indices (2, E) where edge_index[0] = source,
                       edge_index[1] = target.
            edge_type: Edge type indices (E,), values in [0, num_relation_types).

        Returns:
            Updated node features (N, out_dim).
        """
        num_nodes = node_features.size(0)
        output = torch.zeros(num_nodes, self.out_dim, device=node_features.device)

        # Self-loop contribution
        output = output + self.self_loop(node_features)

        # Message passing per relation type
        if edge_index.size(1) > 0:
            for r in range(self.num_relation_types):
                # Find edges of this relation type
                mask = edge_type == r
                if mask.sum() == 0:
                    continue

                r_edge_index = edge_index[:, mask]
                source_nodes = r_edge_index[0]
                target_nodes = r_edge_index[1]

                # Transform source node features with relation-specific weights
                source_features = node_features[source_nodes]
                messages = self.relation_weights[r](source_features)

                # Aggregate messages at target nodes (sum aggregation)
                output.index_add_(0, target_nodes, messages)

        # Normalization: divide by node degree + 1 (for self-loop)
        if edge_index.size(1) > 0:
            target_nodes_all = edge_index[1]
            degree = torch.zeros(num_nodes, device=node_features.device)
            degree.index_add_(
                0, target_nodes_all,
                torch.ones(target_nodes_all.size(0), device=node_features.device),
            )
            degree = degree + 1  # Add 1 for self-loop
            output = output / degree.unsqueeze(1).clamp(min=1)

        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(node_features)
            else:
                residual = node_features
            output = output + residual

        # Layer norm + activation + dropout
        output = self.layer_norm(output)
        output = self.activation(output)
        output = self.dropout(output)

        return output


class HeterogeneousGraphNetwork(nn.Module):
    """Multi-layer heterogeneous graph neural network.

    Embeds anatomical knowledge graph into feature space through L layers
    of heterogeneous graph convolution. Produces anatomically-conditioned
    node embeddings h̃_v = h_v^{(L)} that govern both hallucination and fusion.

    The network takes node features initialized by the text encoder:
        h_v^{(0)} = E_t(T_v), where T_v is the textual description of node v.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_relation_types: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        """Initialize the graph network.

        Args:
            feature_dim: Input/output feature dimension (d).
            hidden_dim: Hidden layer dimension.
            num_layers: Number of graph convolution layers (L).
            num_relation_types: Number of edge relation types.
            dropout: Dropout probability.
            residual: Use residual connections.
        """
        super().__init__()

        self.num_layers = num_layers

        # Build graph convolution layers
        layers = []
        for i in range(num_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else feature_dim
            layers.append(
                HeterogeneousGraphConvLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_relation_types=num_relation_types,
                    dropout=dropout,
                    residual=residual,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute anatomically-conditioned node embeddings.

        Args:
            node_features: Initial node features (N, d), typically from
                          the text encoder.
            edge_index: Edge indices (2, E).
            edge_type: Edge type indices (E,).

        Returns:
            Updated node embeddings h̃ = (N, d) after L layers of
            heterogeneous graph convolution.
        """
        h = node_features

        for layer in self.layers:
            h = layer(h, edge_index, edge_type)

        return h
