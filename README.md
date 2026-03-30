# MedZFS: Hallucinating Anatomical Prototypes for Unified Zero-to-Few-Shot Medical Image Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Conference-ICML%202026-purple" alt="ICML">
</p>

> This repository implements the method proposed in the accompanying research paper submitted to ICML 2026.

---

## Abstract

Medical segmentation faces a fundamental dichotomy: zero-shot methods utilize semantic knowledge but lack adaptability, while few-shot methods require scarce annotations. **MedZFS** is a unified framework bridging this gap via vision-language prototype hallucination and anatomical graph constraints.

Unlike existing approaches, MedZFS:
- **Synthesizes** structurally consistent visual prototypes directly from medical text for zero-shot initialization
- **Refines** prototypes using hard prototype mining within a bilevel optimization framework
- **Integrates** anatomical knowledge through heterogeneous graph message passing

### Key Results

| Setting | Abd-MRI | Abd-CT | CMR |
|---------|---------|--------|-----|
| **Zero-Shot** | 65.8% | 66.5% | 65.0% |
| **1-Shot** | 88.5% | 87.5% | 84.6% |
| **5-Shot** | 90.9% | 90.7% | 90.2% |
| **10-Shot** | 93.3% | 93.4% | 93.1% |

---

## Key Contributions

1. **Vision-Language Prototype Hallucination** — Generates structured distributions of visual prototypes directly from anatomical text descriptions, enabling zero-shot segmentation without any labeled examples.

2. **Anatomical Graph-Constrained Reasoning** — Embeds anatomical knowledge (spatial, hierarchical, functional relations) inside the feature space through heterogeneous graph message passing, achieving 93.8% anatomical constraint satisfaction.

3. **Unified Zero-to-Few-Shot Framework** — Seamlessly operates across all supervision regimes (k=0,1,5,10) via bilevel optimization with 2.8× improved sample efficiency.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │                  MedZFS Framework               │
                    └─────────────────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │   PATHWAY 1:     │     │   PATHWAY 2:     │     │   PATHWAY 3:     │
   │  Hallucination   │     │   Refinement     │     │  Graph Fusion    │
   │  (Always Active) │     │  (k > 0 only)    │     │  (Always Active) │
   └──────────────────┘     └──────────────────┘     └──────────────────┘
          │                          │                          │
   ┌──────┴──────┐           ┌──────┴──────┐           ┌──────┴──────┐
   │ Text Encoder │           │ Visual Enc. │           │ Episode     │
   │ (BioClinBERT)│           │ (ResNet-101)│           │ Graph GNN   │
   └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
          │                          │                          │
   ┌──────┴──────┐           ┌──────┴──────┐                   │
   │ Anatomical  │           │ Hard Proto  │                   │
   │ Graph Conv  │           │ Mining      │                   │
   └──────┬──────┘           └──────┬──────┘                   │
          │                          │                          │
   ┌──────┴──────┐                   │                          │
   │ Gaussian    │                   │                          │
   │ Sampling    │                   │                          │
   └──────┬──────┘                   │                          │
          │                          │                          │
          │    P_halluc              │    P_hard                │
          └──────────────┬───────────┘                          │
                         │                                      │
                         ▼                                      │
                  ┌──────────────┐                              │
                  │ Fused Proto  │◄─────────────────────────────┘
                  │ Set P_c      │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Cosine Sim   │
                  │ Segmentation │
                  └──────┬───────┘
                         │
                         ▼
                    ┌──────────┐
                    │  Ŷ_q     │
                    └──────────┘
```

---

## Repository Structure

```
MedZFS/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
├── setup.py                   # Package installation
├── .gitignore                 # Git ignore rules
├── CITATION.cff               # Citation metadata
│
├── configs/                   # Experiment configurations
│   ├── config.yaml            # Base configuration
│   ├── train_abd_mri.yaml     # Abdomen MRI training
│   ├── train_abd_ct.yaml      # Abdomen CT training
│   ├── train_cmr.yaml         # Cardiac MRI training
│   └── eval.yaml              # Evaluation configuration
│
├── data/                      # Data loading and processing
│   ├── dataset.py             # Episodic medical dataset
│   ├── preprocessing.py       # Volume preprocessing
│   ├── anatomical_graphs.py   # Knowledge graph definitions
│   └── transforms.py          # Domain-specific augmentations
│
├── models/                    # Model architecture
│   ├── medzfs.py              # Full MedZFS model
│   ├── visual_encoder.py      # ResNet-101 backbone
│   ├── text_encoder.py        # BioClinicalBERT encoder
│   ├── hallucination.py       # Prototype hallucination
│   ├── graph_network.py       # Heterogeneous graph conv
│   ├── fusion.py              # Graph-constrained fusion
│   ├── prototype_mining.py    # Hard prototype mining
│   └── loss_functions.py      # All training losses
│
├── training/                  # Training pipeline
│   ├── train.py               # Main entry point
│   ├── trainer.py             # 3-stage bilevel trainer
│   └── scheduler.py           # Learning rate scheduling
│
├── inference/                 # Inference pipeline
│   └── predict.py             # Zero/few-shot prediction
│
├── evaluation/                # Evaluation pipeline
│   ├── metrics.py             # Dice, HD, ACR metrics
│   └── evaluate.py            # Full evaluation driver
│
├── utils/                     # Utility modules
│   ├── logger.py              # TensorBoard/W&B logging
│   ├── seed.py                # Reproducibility utilities
│   └── visualization.py       # Segmentation visualization
│
├── scripts/                   # Shell scripts
│   ├── download_datasets.sh   # Download datasets
│   └── run_training.sh        # Full training pipeline
│
├── docs/                      # Documentation
│   ├── architecture.md        # Architecture details
│   ├── methodology.md         # Algorithmic details
│   └── experiments.md         # Reproducibility guide
│
└── tests/                     # Unit tests
    ├── test_model.py          # Model tests
    ├── test_hallucination.py  # Hallucination tests
    └── test_metrics.py        # Metrics tests
```

---

## Installation

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/anonymous/MedZFS.git
cd MedZFS
conda env create -f environment.yml
conda activate medzfs
```

### Option 2: pip

```bash
git clone https://github.com/anonymous/MedZFS.git
cd MedZFS
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Preparation

MedZFS is evaluated on three benchmarks:

| Dataset | Modality | Organs | Source |
|---------|----------|--------|--------|
| **Abd-MRI** | MRI | Liver, R-Kidney, L-Kidney, Spleen | [CHAOS Challenge](https://chaos.grand-challenge.org/) |
| **Abd-CT** | CT | Spleen, R-Kidney, L-Kidney, Liver | [Synapse Multi-Organ](https://www.synapse.org/) |
| **CMR** | Cardiac MRI | LV-MYO, LV-BP, RV | [MS-CMRSeg](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg/) |

### Download

```bash
bash scripts/download_datasets.sh --dataset all --output_dir ./data/raw/
```

### Preprocessing

```bash
python -m data.preprocessing \
    --input_dir ./data/raw/ \
    --output_dir ./data/processed/ \
    --target_spacing 1.0 1.0 1.0 \
    --normalize
```

---

## Training

### Quick Start (Abdomen MRI)

```bash
python -m training.train --config configs/train_abd_mri.yaml
```

### Full 3-Stage Training Pipeline

```bash
# Stage 1: Zero-shot pre-training (hallucination + graph alignment)
python -m training.train \
    --config configs/train_abd_mri.yaml \
    --stage 1 \
    --lr 1e-4 \
    --epochs 50

# Stage 2: Episodic meta-learning (hard prototype mining)
python -m training.train \
    --config configs/train_abd_mri.yaml \
    --stage 2 \
    --lr 1e-3 \
    --epochs 30 \
    --resume checkpoints/stage1_best.pth

# Stage 3: Joint bilevel optimization
python -m training.train \
    --config configs/train_abd_mri.yaml \
    --stage 3 \
    --lr 5e-5 \
    --epochs 100 \
    --resume checkpoints/stage2_best.pth
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 -m training.train \
    --config configs/train_abd_mri.yaml \
    --distributed
```

### Run All Datasets

```bash
bash scripts/run_training.sh
```

---

## Evaluation

### Evaluate on All Datasets

```bash
python -m evaluation.evaluate \
    --config configs/eval.yaml \
    --checkpoint checkpoints/best_model.pth \
    --shots 0 1 5 10 \
    --datasets abd_mri abd_ct cmr
```

### Zero-Shot Only

```bash
python -m evaluation.evaluate \
    --config configs/eval.yaml \
    --checkpoint checkpoints/best_model.pth \
    --shots 0
```

---

## Inference

### Zero-Shot Prediction

```bash
python -m inference.predict \
    --image path/to/image.nii.gz \
    --target_class liver \
    --checkpoint checkpoints/best_model.pth \
    --output_dir results/
```

### Few-Shot Prediction

```bash
python -m inference.predict \
    --image path/to/query.nii.gz \
    --target_class liver \
    --support_images path/to/support1.nii.gz path/to/support2.nii.gz \
    --support_masks path/to/mask1.nii.gz path/to/mask2.nii.gz \
    --checkpoint checkpoints/best_model.pth \
    --output_dir results/
```

---

## Expected Results

### Zero-Shot Segmentation (Mean Dice %)

| Method | Abd-MRI | Abd-CT | CMR |
|--------|---------|--------|-----|
| MedCLIP | 37.9 | 38.5 | 38.9 |
| MedSAM | 42.5 | 44.3 | 43.7 |
| BiomedCLIP | 45.7 | 47.3 | 47.3 |
| LLaVA-Med | 48.6 | 50.2 | 50.2 |
| **MedZFS (Ours)** | **65.8** | **66.5** | **65.0** |

### Few-Shot Segmentation (Mean Dice %)

| Method | 1-Shot | 5-Shot | 10-Shot |
|--------|--------|--------|---------|
| CoW | 82.2 | 82.5 | 86.2 |
| GMRD | 80.2 | 80.2 | 84.2 |
| **MedZFS (Ours)** | **86.9** | **90.6** | **93.3** |

---

## Reproducibility

- All experiments use a fixed random seed (`--seed 42`)
- Mixed-precision training with gradient scaling
- Exact hyperparameters are specified in config files
- See [docs/experiments.md](docs/experiments.md) for detailed reproducibility instructions

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{anonymous2026medzfs,
    title={MedZFS: Hallucinating Anatomical Prototypes for Unified Zero-to-Few-Shot Medical Image Segmentation},
    author={Anonymous},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This repository builds upon open-source tools and public medical imaging datasets. We thank the maintainers of PyTorch, Hugging Face Transformers, and PyTorch Geometric for their excellent frameworks.
