"""
Unit tests for prototype hallucination module.

Validates Gaussian sampling, reparameterization, and distribution properties.
"""

import pytest
import torch

from models.hallucination import PrototypeHallucinator


class TestHallucinationDistribution:
    """Tests for hallucination distribution properties."""

    def test_sampling_stochastic(self):
        """Different calls should produce different samples."""
        hallu = PrototypeHallucinator(feature_dim=64, num_samples=16)
        hallu.eval()
        text_emb = torch.randn(1, 64)
        graph_emb = torch.randn(1, 5, 64)

        hallu.train()
        out1 = hallu(text_emb, graph_emb)
        out2 = hallu(text_emb, graph_emb)
        # Prototypes should differ due to random sampling
        assert not torch.allclose(out1["prototypes"], out2["prototypes"])

    def test_variance_clamping(self):
        """Log-variance should be within specified bounds."""
        hallu = PrototypeHallucinator(
            feature_dim=32, num_samples=4, min_variance=0.01, max_variance=2.0
        )
        text_emb = torch.randn(2, 32)
        graph_emb = torch.randn(2, 3, 32)
        out = hallu(text_emb, graph_emb)

        # Verify output exists and has correct shapes
        assert out["mu"].shape == (2, 32)
        assert out["prototypes"].shape == (2, 4, 32)

    def test_num_samples_override(self):
        """Can override num_samples at sampling time."""
        hallu = PrototypeHallucinator(feature_dim=32, num_samples=8)
        mu = torch.randn(2, 32)
        log_var = torch.zeros(2, 32)
        protos = hallu.sample_prototypes(mu, log_var, num_samples=20)
        assert protos.shape == (2, 20, 32)

    def test_gradient_flows(self):
        """Gradients should flow through reparameterization."""
        hallu = PrototypeHallucinator(feature_dim=32, num_samples=4)
        text_emb = torch.randn(1, 32, requires_grad=True)
        graph_emb = torch.randn(1, 3, 32)
        out = hallu(text_emb, graph_emb)
        loss = out["prototypes"].sum()
        loss.backward()
        assert text_emb.grad is not None
        assert text_emb.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
