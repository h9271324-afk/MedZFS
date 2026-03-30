"""
Unit tests for MedZFS model architecture.

Validates forward pass shapes, module composition, and zero-shot vs few-shot
pathway activation.
"""

import pytest
import torch

from models.visual_encoder import VisualEncoder
from models.text_encoder import TextEncoder
from models.graph_network import HeterogeneousGraphNetwork
from models.hallucination import PrototypeHallucinator
from models.fusion import GraphConstrainedFusion
from models.prototype_mining import HardPrototypeMiner


class TestVisualEncoder:
    """Tests for the ResNet-101 visual encoder."""

    def test_output_shape(self):
        encoder = VisualEncoder(feature_dim=512, pretrained="none")
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.dim() == 4
        assert out.shape[0] == 2
        assert out.shape[1] == 512

    def test_l2_normalized(self):
        encoder = VisualEncoder(feature_dim=256, pretrained="none")
        x = torch.randn(1, 3, 128, 128)
        out = encoder(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestGraphNetwork:
    """Tests for the heterogeneous graph network."""

    def test_output_shape(self):
        gnn = HeterogeneousGraphNetwork(feature_dim=64, hidden_dim=64, num_layers=2)
        node_features = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        out = gnn(node_features, edge_index, edge_type)
        assert out.shape == (5, 64)

    def test_empty_edges(self):
        gnn = HeterogeneousGraphNetwork(feature_dim=32, num_layers=1)
        node_features = torch.randn(3, 32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        out = gnn(node_features, edge_index, edge_type)
        assert out.shape == (3, 32)


class TestPrototypeHallucinator:
    """Tests for prototype hallucination."""

    def test_output_shape(self):
        hallu = PrototypeHallucinator(feature_dim=64, num_samples=8)
        text_emb = torch.randn(2, 64)
        graph_emb = torch.randn(2, 5, 64)
        out = hallu(text_emb, graph_emb)
        assert out["prototypes"].shape == (2, 8, 64)
        assert out["mu"].shape == (2, 64)
        assert out["log_var"].shape == (2, 64)

    def test_prototypes_normalized(self):
        hallu = PrototypeHallucinator(feature_dim=64, num_samples=4)
        text_emb = torch.randn(1, 64)
        graph_emb = torch.randn(1, 3, 64)
        out = hallu(text_emb, graph_emb)
        norms = out["prototypes"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestFusion:
    """Tests for graph-constrained fusion."""

    def test_zero_shot(self):
        fusion = GraphConstrainedFusion(feature_dim=64)
        hall_protos = torch.randn(2, 8, 64)
        empty_hard = torch.zeros(2, 0, 64)
        graph_emb = torch.randn(2, 5, 64)
        out = fusion(hall_protos, empty_hard, graph_emb)
        assert out.shape == (2, 8, 64)

    def test_few_shot(self):
        fusion = GraphConstrainedFusion(feature_dim=64)
        hall_protos = torch.randn(2, 8, 64)
        hard_protos = torch.randn(2, 4, 64)
        graph_emb = torch.randn(2, 5, 64)
        out = fusion(hall_protos, hard_protos, graph_emb)
        assert out.shape == (2, 12, 64)

    def test_similarity_map(self):
        fusion = GraphConstrainedFusion(feature_dim=64)
        features = torch.randn(2, 64, 16, 16)
        protos = torch.randn(2, 8, 64)
        sim_map = fusion.compute_similarity_map(features, protos)
        assert sim_map.shape == (2, 16, 16)


class TestHardPrototypeMiner:
    """Tests for hard prototype mining."""

    def test_mine_shape(self):
        miner = HardPrototypeMiner(feature_dim=64, num_hard_prototypes=4)
        support_feat = torch.randn(2, 3, 64, 16, 16)
        support_mask = torch.ones(2, 3, 16, 16)
        zs_pred = torch.zeros(2, 3, 16, 16)  # All wrong → all hard
        out = miner(support_feat, support_mask, zs_pred)
        assert out.shape == (2, 12, 64)  # 3 supports × 4 hard each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
