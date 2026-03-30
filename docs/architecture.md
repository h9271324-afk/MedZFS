# MedZFS Architecture

This document describes the detailed architecture of MedZFS.

## Overview

MedZFS is a three-pathway architecture for unified zero-to-few-shot medical image segmentation:

1. **Hallucination Pathway** — Generates visual prototypes from text
2. **Refinement Pathway** — Mines hard prototypes from support examples
3. **Graph-Constrained Fusion** — Integrates all prototypes

## Component Details

### Visual Encoder

- **Backbone:** ResNet-101 initialized from BiomedCLIP weights
- **Feature Extraction:** Multi-scale features from blocks 2, 3, 4
- **Fusion:** FPN-style top-down pathway with lateral connections
- **Output:** Dense feature map F_q ∈ R^{B×d×h×w}, L2-normalized
- **Feature dimension:** d = 512

### Text Encoder

- **Backbone:** BioClinicalBERT (768-dim)
- **Pooling:** [CLS] token representation
- **Projection:** MLP (768 → 512) with LayerNorm and GELU
- **Output:** z_c ∈ R^d, L2-normalized
- **Freezing:** Frozen during stages 1-2, unfrozen at stage 3

### Anatomical Graph Network

- **Type:** Heterogeneous graph convolution
- **Layers:** L = 4 with residual connections
- **Relation types:** spatial, hierarchical, functional, pathological
- **Edge computation:** W_r^{(l)} · h_u^{(l)} per relation type
- **Aggregation:** Sum + degree normalization + self-loop
- **Initialization:** Node features from text encoder

### Prototype Hallucinator

- **Distribution:** Gaussian N(μ_c, Σ_c)
- **Conditioning:** Cross-attention between text embedding and graph embeddings
- **μ network:** MultiheadAttention + LayerNorm + MLP
- **Σ network:** Separate attention head → log-variance
- **Sampling:** Reparameterization trick, M = 16 prototypes
- **Variance bounds:** [0.01, 2.0]

### Hard Prototype Miner

- **Activation:** Only for k > 0 (few-shot)
- **Hard region detection:** Pixels where GT=1 but hallucination predicts 0
- **Mining:** Stochastic subsampling (50% ratio) + mean pooling
- **Alignment:** Learned projection Π for cross-modal consistency
- **Output:** 8 hard prototypes per support image

### Graph-Constrained Fusion

- **Architecture:** 2-layer transformer with:
  - Cross-attention: prototypes → anatomical embeddings
  - Self-attention: prototypes ↔ prototypes
  - FFN: 4× expansion ratio
- **Output:** Fused prototype set P̃_c, L2-normalized

### Segmentation Head

- **Similarity:** Cosine similarity with temperature τ = 0.1
- **Aggregation:** Max over all prototypes (Eq. 4)
- **Upsampling:** Bilinear interpolation to input resolution
- **Activation:** Sigmoid for probability map

## Dynamic Pathway Activation

| Component | k=0 | k=1 | k=5 | k=10 |
|-----------|-----|-----|-----|------|
| Visual Encoder | ✓ | ✓ | ✓ | ✓ |
| Text Encoder | ✓ | ✓ | ✓ | ✓ |
| Graph Network | ✓ | ✓ | ✓ | ✓ |
| Hallucinator | ✓ | ✓ | ✓ | ✓ |
| Hard Miner | ✗ | ✓ | ✓ | ✓ |
| Fusion | ✓ | ✓ | ✓ | ✓ |

## Parameter Count

| Module | Parameters |
|--------|-----------|
| Visual Encoder (ResNet-101) | ~44M |
| Text Encoder (BioClinicalBERT) | ~110M |
| Graph Network (4 layers) | ~4M |
| Hallucinator | ~8M |
| Hard Miner | ~0.5M |
| Fusion (2 layers) | ~8M |
| **Total** | **~175M** |
