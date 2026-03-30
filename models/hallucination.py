"""
Prototype Hallucination Network for MedZFS.

Implements the Graph-Constrained Prototype Hallucination module (Section 4.2)
that generates visual prototypes from anatomical text descriptions without
requiring any labeled images (zero-shot).

Models prototypes as Gaussian distributions:
    p_c ~ N(μ_c, Σ_c)           (Eq. 5)
    μ_c = g_μ(z_c, H̃)          (Eq. 14)
    Σ_c = g_Σ(z_c, H̃)          (Eq. 14)
    p_c^j = μ_c + Σ_c^{1/2} ε_j (Eq. 15)  [Reparameterization trick]
"""

import torch
import torch.nn as nn


class PrototypeHallucinator(nn.Module):
    """Hallucinate visual prototypes from text and anatomical graph embeddings.

    Generates a distribution of M visual prototypes conditioned on:
      1. Text embedding z_c from the target class description
      2. Graph-conditioned anatomical embeddings H̃ from the graph network

    The hallucinated prototypes P_c^{(0)} serve as zero-shot anchors for
    segmentation, enabling predictions without any labeled examples.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_samples: int = 16,
        min_variance: float = 0.01,
        max_variance: float = 2.0,
    ):
        """Initialize the hallucination network.

        Args:
            feature_dim: Feature space dimension (d).
            num_samples: Number of prototypes to sample (M).
            min_variance: Minimum variance for numerical stability.
            max_variance: Maximum variance clamp to prevent explosion.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.min_variance = min_variance
        self.max_variance = max_variance

        # g_μ: Maps (z_c, H̃) → μ_c
        #   Combines text embedding with graph-conditioned context via
        #   cross-attention followed by a projection MLP.
        self.mu_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.mu_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # g_Σ: Maps (z_c, H̃) → log(diag(Σ_c))
        #   Predicts log-variance for numerical stability.
        self.sigma_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.sigma_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # Layer norms for stability
        self.mu_norm = nn.LayerNorm(feature_dim)
        self.sigma_norm = nn.LayerNorm(feature_dim)

    def compute_distribution_params(
        self, text_embedding: torch.Tensor, graph_embeddings: torch.Tensor
    ) -> tuple:
        """Compute the mean and log-variance of the prototype distribution.

        Uses cross-attention where the text embedding queries the
        graph-conditioned anatomical embeddings to produce distribution
        parameters that respect anatomical structure.

        Args:
            text_embedding: Text embedding z_c of shape (B, d) or (B, 1, d).
            graph_embeddings: Anatomical node embeddings H̃ of shape (B, N, d).

        Returns:
            Tuple of (mu, log_var) each of shape (B, d).
        """
        # Ensure text embedding has sequence dimension
        if text_embedding.dim() == 2:
            text_embedding = text_embedding.unsqueeze(1)  # (B, 1, d)

        # Cross-attention for mean: text queries graph embeddings
        mu_attn_out, _ = self.mu_attention(
            query=text_embedding,
            key=graph_embeddings,
            value=graph_embeddings,
        )
        mu = self.mu_norm(text_embedding + mu_attn_out)
        mu = self.mu_projection(mu.squeeze(1))  # (B, d)

        # Cross-attention for variance: separate attention head
        sigma_attn_out, _ = self.sigma_attention(
            query=text_embedding,
            key=graph_embeddings,
            value=graph_embeddings,
        )
        log_var = self.sigma_norm(text_embedding + sigma_attn_out)
        log_var = self.sigma_projection(log_var.squeeze(1))  # (B, d)

        return mu, log_var

    def sample_prototypes(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        num_samples: int = None,
    ) -> torch.Tensor:
        """Sample prototypes using the reparameterization trick.

        Implements Eq. 15: p_c^j = μ_c + Σ_c^{1/2} · ε_j, ε_j ~ N(0, I)

        Args:
            mu: Mean vector (B, d).
            log_var: Log-variance vector (B, d).
            num_samples: Number of prototypes M to sample. Defaults to
                        self.num_samples.

        Returns:
            Sampled prototypes of shape (B, M, d).
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Clamp log-variance for stability
        log_var = torch.clamp(
            log_var,
            min=torch.log(torch.tensor(self.min_variance, device=log_var.device)),
            max=torch.log(torch.tensor(self.max_variance, device=log_var.device)),
        )

        # Compute standard deviation
        std = torch.exp(0.5 * log_var)  # (B, d)

        # Sample noise: ε ~ N(0, I)
        epsilon = torch.randn(
            mu.size(0), num_samples, self.feature_dim, device=mu.device
        )  # (B, M, d)

        # Reparameterization trick
        mu_expanded = mu.unsqueeze(1).expand(-1, num_samples, -1)    # (B, M, d)
        std_expanded = std.unsqueeze(1).expand(-1, num_samples, -1)  # (B, M, d)

        prototypes = mu_expanded + std_expanded * epsilon  # (B, M, d)

        # L2 normalize prototypes for cosine similarity
        prototypes = nn.functional.normalize(prototypes, p=2, dim=-1)

        return prototypes

    def forward(
        self,
        text_embedding: torch.Tensor,
        graph_embeddings: torch.Tensor,
    ) -> dict:
        """Hallucinate prototypes from text and anatomical graph.

        Args:
            text_embedding: Text embedding z_c of shape (B, d).
            graph_embeddings: Graph node embeddings H̃ of shape (B, N, d).

        Returns:
            Dictionary containing:
              - prototypes: Hallucinated prototypes P_c^{(0)} of shape (B, M, d)
              - mu: Mean vector (B, d)
              - log_var: Log-variance vector (B, d)
        """
        # Compute distribution parameters conditioned on text + anatomy
        mu, log_var = self.compute_distribution_params(
            text_embedding, graph_embeddings
        )

        # Sample M prototypes via reparameterization
        prototypes = self.sample_prototypes(mu, log_var)

        return {
            "prototypes": prototypes,
            "mu": mu,
            "log_var": log_var,
        }
