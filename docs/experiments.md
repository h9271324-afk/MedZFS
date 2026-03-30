# Experiment Reproducibility Guide

This document provides exact steps to reproduce all results in the paper.

## Hardware Requirements

- **GPU:** 4× NVIDIA A100 (40GB) or equivalent
- **RAM:** 64GB minimum
- **Storage:** 100GB for datasets + checkpoints

## Software Environment

```bash
conda env create -f environment.yml
conda activate medzfs
pip install -e .
```

## Dataset Preparation

### 1. Download Datasets

| Dataset | Source | Registration |
|---------|--------|-------------|
| Abd-MRI (CHAOS) | [chaos.grand-challenge.org](https://chaos.grand-challenge.org/) | Required |
| Abd-CT (Synapse) | [synapse.org](https://www.synapse.org/) | Required |
| CMR (MS-CMRSeg) | [fudan.edu.cn](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg/) | Required |

### 2. Preprocess

```bash
# Abdomen MRI
python -m data.preprocessing --input_dir data/raw/abd_mri --output_dir data/processed/abd_mri --modality MRI

# Abdomen CT
python -m data.preprocessing --input_dir data/raw/abd_ct --output_dir data/processed/abd_ct --modality CT

# Cardiac MRI
python -m data.preprocessing --input_dir data/raw/cmr --output_dir data/processed/cmr --modality CMR
```

## Training

### Full Pipeline (Recommended)

```bash
bash scripts/run_training.sh
```

### Manual Three-Stage Training

```bash
# Stage 1: Zero-shot pre-training (50 epochs)
python -m training.train --config configs/train_abd_mri.yaml --stage 1 --seed 42

# Stage 2: Episodic meta-learning (30 epochs)
python -m training.train --config configs/train_abd_mri.yaml --stage 2 --seed 42 --resume checkpoints/stage1_best.pth

# Stage 3: Joint bilevel optimization (100 epochs)
python -m training.train --config configs/train_abd_mri.yaml --stage 3 --seed 42 --resume checkpoints/stage2_best.pth
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Feature dimension (d) | 512 |
| Graph layers (L) | 4 |
| Hallucinated prototypes (M) | 16 |
| Hard prototypes per support | 8 |
| Stage 1: LR, epochs | 1e-4, 50 |
| Stage 2: LR, epochs | 1e-3, 30 |
| Stage 3: LR, epochs | 5e-5, 100 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Loss: λ₁ (graph) | 0.1 |
| Loss: λ₂ (align) | 0.05 |
| Loss: λ₃ (boundary) | 0.1 |

## Evaluation

```bash
python -m evaluation.evaluate \
    --config configs/eval.yaml \
    --checkpoint checkpoints/stage3_best.pth \
    --shots 0 1 5 10 \
    --datasets abd_mri abd_ct cmr
```

## Expected Results

### Zero-Shot (Table 1)

| Method | Abd-MRI | Abd-CT | CMR |
|--------|---------|--------|-----|
| **MedZFS** | **65.8** | **66.5** | **65.0** |

### Few-Shot (Table 2)

| Shots | Abd-MRI | Abd-CT | CMR |
|-------|---------|--------|-----|
| 1 | 88.5 | 87.5 | 84.6 |
| 5 | 90.9 | 90.7 | 90.2 |
| 10 | 93.3 | 93.4 | 93.1 |

## Troubleshooting

- **OOM errors:** Reduce batch_size in config or use fewer GPUs
- **Slow training:** Ensure mixed_precision is enabled
- **Poor zero-shot:** Verify BioClinicalBERT weights are loaded correctly
- **NaN losses:** Check that graph edges are valid and variance bounds are set
