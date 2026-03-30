# MedZFS Methodology

This document describes the mathematical foundations and algorithms of MedZFS.

## Problem Formulation

Given a query image I_q, anatomical text T_c, knowledge graph G_c, and support set S_k (k ≥ 0), learn a predictor:

```
f_Θ : (I_q, T_c, G_c, S_k) → Ŷ_q
```

## Algorithm Overview

### Algorithm 1: MedZFS Inference

```
Input: Query image I_q, text T_c, graph G_c, support S_k
Output: Segmentation mask Ŷ_q

1. Encode anatomy: H̃ = GraphNetwork(TextEncoder(G_c.nodes), G_c.edges)
2. Encode text: z_c = TextEncoder(T_c)
3. Hallucinate: (μ_c, Σ_c) = Hallucinator(z_c, H̃)
4. Sample M prototypes: P_c^{(0)} = {μ_c + Σ_c^{1/2}·ε_j}_{j=1}^M
5. Encode query: F_q = VisualEncoder(I_q)
6. IF k > 0:
   a. Encode support: F_i = VisualEncoder(S_k.images)
   b. Zero-shot predict: Ŷ_i^{(0)} = Segment(F_i, P_c^{(0)})
   c. Find hard regions: Ω_hf = {x | Y_i(x)=1, Ŷ_i^{(0)}(x)=0}
   d. Mine hard prototypes: P_hard = MeanPool(F_i[Ω_hf])
7. Fuse: P̃_c = GraphFusion(P_c^{(0)}, P_hard, H̃)
8. Segment: Ŷ_q(x) = σ(max_p cos(F_q(x), p) / τ),  p ∈ P̃_c
```

### Algorithm 2: Three-Stage Training

```
Stage 1: Zero-Shot Pre-Training (50 epochs, lr=1e-4)
  - Train hallucinator + graph network on image-text pairs
  - Freeze text encoder backbone
  - Loss: L_seg + λ₁·L_graph

Stage 2: Episodic Meta-Learning (30 epochs, lr=1e-3)
  - Enable hard prototype mining
  - Episodic training with varying k
  - Loss: L_seg + λ₁·L_graph + λ₂·L_align

Stage 3: Joint Bilevel Optimization (100 epochs, lr=5e-5)
  - Unfreeze all parameters including text encoder
  - Bilevel: upper=zero-shot, lower=few-shot objectives
  - Gradient surgery for conflicting gradients
  - Loss: L_seg + λ₁·L_graph + λ₂·L_align + λ₃·L_boundary
```

## Loss Functions

### Total Loss (Eq. 20)
```
L = L_seg + λ₁·L_graph + λ₂·L_align + λ₃·L_boundary
```

| Loss | Weight | Description |
|------|--------|-------------|
| L_seg | 1.0 | BCE + Dice segmentation loss |
| L_graph | λ₁=0.1 | Anatomical relation consistency |
| L_align | λ₂=0.05 | Hallucinated-visual prototype alignment |
| L_boundary | λ₃=0.1 | Distance-transform boundary loss |

## Theoretical Foundations

### Prototype Hallucination (Eq. 5-6)
Prototypes are modeled as Gaussian random variables conditioned on text and anatomical structure. The reparameterization trick enables end-to-end gradient flow through sampling.

### Graph Constraints (Eq. 16)
The graph consistency loss enforces that hallucinated prototypes respect anatomical spatial relations by minimizing the displacement from learned relation embeddings δ_r.

### Bilevel Optimization (Eq. 11-12)
The upper level optimizes zero-shot hallucination quality; the lower level optimizes few-shot adaptation. Gradient surgery prevents conflicts between these objectives.
